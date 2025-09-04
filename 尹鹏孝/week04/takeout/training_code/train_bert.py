import pandas as pd
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from logger import logger

# -------------------------- 1. 数据准备 --------------------------
# 加载数据集，指定分隔符为制表符，并无表头
dataset = pd.read_csv("../assets/dataset/作业数据-waimai_10k-change.csv", header=None,skiprows=[0])


# 初始化并拟合标签编码器，将文本标签（如“体育”）转换为数字标签（如0, 1, 2...）
lbl = LabelEncoder()
logger.info('初始拟合标签编码：')
logger.info(dataset[0].values)
# 取出800样本
nums_len = 800
lbl.fit(dataset[0].values)
labels_init = lbl.transform(dataset[0].values[:nums_len])
texts_init = list(dataset[1].values[:nums_len])
# lbl.fit_transform(dataset[1].values[:nums_len])

# 将数据按8:2的比例分割为训练集和测试集
# stratify 参数确保训练集和测试集中各类别的样本比例与原始数据集保持一致
x_train, x_test, train_label, test_label = train_test_split(

    texts_init,# 文本数据
    labels_init,# 对应的数字标签
    test_size=0.2,# 测试集比例为20%
    stratify=dataset[0][:nums_len].values# 确保训练集和测试集的标签分布一致
)

# 加载BERT预训练的分词器（Tokenizer）
# 分词器负责将文本转换为模型可识别的输入ID、注意力掩码等
tokenizer = BertTokenizer.from_pretrained('../assets/models/google-bert/bert-base-chinese')

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

# 使用DataLoader创建批量数据加载器
# batch_size=16：每个批次包含16个样本
# shuffle=True：在每个epoch开始时打乱数据，以提高模型泛化能力
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
# -------------------------- 3. 模型和优化器 --------------------------
# 加载BERT用于序列分类的预训练模型
# num_labels=17：指定分类任务的类别数量
# BertForSequenceClassification全连接文本分类
model = BertForSequenceClassification.from_pretrained('../assets/models/google-bert/bert-base-chinese', num_labels=16)

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


# -------------------------- 4. 训练和验证函数 --------------------------
# 定义训练函数
def train():
    # 设置模型为训练模式
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)

    # 遍历训练数据加载器
    for batch in train_loader:
        # 清除上一轮的梯度
        optim.zero_grad()

        # 将批次数据移动到指定设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 执行前向传播，得到损失和logits
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()

        # 反向传播计算梯度
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新模型参数
        optim.step()

        iter_num += 1
        # 每100步打印一次训练进度
        if (iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    # 打印平均训练损失
    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))


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

for epoch in range(20):
    print("------------Epoch: %d ----------------" % epoch)
    # 训练模型
    train()
    # 验证模型
    validation()
# -------------------------- 6. 输出模型 --------------------------
# 输出模型供model模块使用
torch.save(model.state_dict(), '../assets/weights/bert.pkl')



