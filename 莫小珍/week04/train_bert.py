import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# -------------------------- 1. 数据准备 --------------------------
# 加载数据集，指定分隔符为制表符，并无表头
dataset = pd.read_csv("../作业数据-waimai_10k.csv", sep=",")

label = dataset['label'].to_list()
text = dataset['review'].to_list()


##数据集没有打乱，前面的标签都是1，后面的标签都是0
text_new = text[:250]+text[-250:]
label_new = label[:250]+label[-250:]


# 分割数据为训练集和测试集
x_train, x_test, train_label, test_label = train_test_split(
    text_new,             # 文本数据 包含所有文本数据的数组或列表
    label_new,            # 对应的数字标签 与文本数据对应的数字标签数组或列表
    test_size=0.2,     # 测试集比例为20%
#   stratify=label_new    # 确保训练集和测试集的标签分布一致  确保训练集和测试集中各类别的比例与原始数据集相同
)

# 加载BERT预训练的分词器（Tokenizer）
# 分词器负责将文本转换为模型可识别的输入ID、注意力掩码等
tokenizer = BertTokenizer.from_pretrained('../../models/google-bert/bert-base-chinese')

# 对训练集和测试集的文本进行编码
# truncation=True：如果句子长度超过max_length，则截断
# padding=True：将所有句子填充到max_length
# max_length=64：最大序列长度
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# -------------------------- 2. 数据集和数据加载器 --------------------------
# 自定义数据集类，继承自PyTorch的Dataset
# 用于处理编码后的数据和标签，方便后续批量读取
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels


    # 获取单个样本的方法
    def __getitem__(self, idx):
        # 从编码字典中提取input_ids, attention_mask等，并转换为PyTorch张量
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 添加标签，并转换为张量
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item



    # 返回数据集总样本数的方法
    def __len__(self):
        return len(self.labels)


# 实例化自定义数据集
train_dataset = NewsDataset(train_encoding, train_label)
test_dataset = NewsDataset(test_encoding, test_label)


# import pdb; pdb.set_trace()  手工打断点

# 使用DataLoader创建批量数据加载器
# batch_size=16：每个批次包含16个样本
# shuffle=True：在每个epoch开始时打乱数据，以提高模型泛化能力
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)


# -------------------------- 3. 模型和优化器 --------------------------
# 加载BERT用于序列分类的预训练模型
# num_labels=17：指定分类任务的类别数量
model = BertForSequenceClassification.from_pretrained('../../models/google-bert/bert-base-chinese', num_labels=17)

# 设置设备，优先使用CUDA（GPU），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到指定的设备上
model.to(device)

# 定义优化器，使用AdamW，lr是学习率
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 定义精度计算函数
def flat_accuracy(preds, labels):
    # 获取预测结果的最高概率索引
    pred_flat = np.argmax(preds, axis=1).flatten()
    # 展平真实标签
    labels_flat = labels.flatten()
    # 计算准确率
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


#4.训练和验证函数
def get_accuracy(logits,labels):
    predicts = np.argmax(logits,axis=1).flatten()
    real_labels = labels.flatten()
    return np.sum(predicts == real_labels) / len(real_labels)

def train():
    model.train()
    total_train_loass = 0
    iter_num = 0
    total_iter = len(train_loader)

    for batch_data in train_loader:
        #清除上一轮梯度
        optim.zero_grad()

        #将批次数据移动到指定设备
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        labels = batch_data['labels'].to(device)

        #前向传播
        outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
        loss = outputs[0]
        total_train_loass += loss.item()

        #反向传播计算梯度
        loss.backward()
        #梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=1.0)

        #更新参数
        optim.step()

        iter_num += 1
        if (iter_num % 10 == 0):
            print("epoch: %d,iter_num: %d, loss: %.4f ,%.2f%%" % (epoch,iter_num,loss.item(),iter_num/total_iter*100))

    #平均训练损失
    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loass / len(train_loader)))

# 定义验证函数
def validation():
    # 设置模型为评估模式
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    # 遍历测试数据加载器
    for batch in test_dataloader:
        # 在验证阶段，不计算梯度
        with torch.no_grad():
            # 将批次数据移动到指定设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 执行前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        # 将logits和标签从GPU移动到CPU，并转换为numpy数组
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # 计算平均准确率
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))
    print("-------------------------------")


# -------------------------- 5. 主训练循环 --------------------------
# 循环训练4个epoch
for epoch in range(4):
    print("------------Epoch: %d ----------------" % epoch)
    # 训练模型
    train()
    # 验证模型
    validation()

