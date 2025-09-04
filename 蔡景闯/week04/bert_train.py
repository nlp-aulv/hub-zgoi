import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# 1. 数据集准备
data = pd.read_csv('./作业数据-waimai_10k.csv', header=0)
labels = data['label'].tolist()
reviews = data['review'].tolist()  # 转换为字符串类型，分词器要求必须是字符串类型

# 读取BERT的分词器
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
# 划分训练集和测试集
x_train, x_test, label_train, label_test = train_test_split(reviews,  # 文本数据
                                                            labels,  # 标签数据
                                                            test_size=0.2,
                                                            stratify=labels)
# 对训练集和测试集的文本进行编码
# truncation=True：如果句子长度超过max_length，则截断
# padding=True：将所有句子填充到max_length
# max_length=64：最大序列长度

# 此处分词器要求必须是字符串或者字符串列表类型
# 分词器的输出是一个字典，包含input_ids（文本转换成的 token ID 序列）, attention_mask（模型应关注哪些位置的标记），token_type_ids（区分不同句子的标记）
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=128)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=128)


# 定义数据集
class MyDataset(Dataset):
    def __init__(self, labels1, encodings):
        self.encodings = encodings
        self.labels = labels1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 从编码字典中提取input_ids, attention_mask等，并转换为PyTorch张量
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 添加标签，并转换为张量
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item


# 准备训练和测试数据集
dataset_train = MyDataset(label_train, train_encoding)
dataset_test = MyDataset(label_test, test_encoding)
train_loader = DataLoader(dataset_train, batch_size=20, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=20, shuffle=True)

# 2. 模型准备
# 读取BERT预训练模型
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese',
                                                      num_labels=2)  # 数据中的标签只有两种，0和1
# 设置设备，优先使用CUDA（GPU），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到指定的设备上
model.to(device)
# 定义优化器，使用AdamW，lr是学习率
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)


# 3. 训练模型
def train(epochs):
    model.train()

    for epoch in range(epochs):
        total_train_loss = 0
        iter_num = 0  # 记录每轮训练次数
        for batch in train_loader:
            # 清除上一轮的梯度
            optim.zero_grad()

            # 将批次数据移动到指定设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels2 = batch['labels'].to(device)

            # 执行前向传播，得到损失和logits
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels2)
            loss = outputs[0]
            total_train_loss += loss.item()

            # 反向传播计算梯度
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新模型参数
            optim.step()

            iter_num += 1
            # 每50步打印一次训练进度
            if iter_num % 100 == 0:
                print(f"epoch: {epoch + 1}/{epochs}, iter_num: {iter_num}, loss: {loss.item():.4f}")

        # 打印每轮的平均训练损失
        print(f"Epoch: {epoch + 1} Average training loss: {(total_train_loss / len(train_loader)):.4f}")
        # 评估模型
        evaluate()

    # 保存模型
    torch.save(model.state_dict(), './bert_model.pth')

def evaluate():
    model.eval()
    total_eval_loss = 0
    total_correct = 0
    total_samples = 0 # 记录总样本数
    with torch.no_grad():
        for batch in test_loader:
            # 将批次数据移动到指定设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels2 = batch['labels'].to(device)

            # 执行前向传播，得到损失和logits
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels2)
            loss = outputs[0]
            logits = outputs[1]

            # 按样本数加权累加损失
            batch_size = input_ids.size(0)
            total_eval_loss += loss.item() * batch_size

            # 计算预测和正确数
            predictions = torch.argmax(logits, dim=1)
            correct = torch.sum(predictions == labels2)
            total_correct += correct
            total_samples += batch_size

            # 计算平均损失和准确率
        avg_loss = total_eval_loss / total_samples
        accuracy = total_correct / total_samples * 100  # 百分比形式

        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")

def predict(text):
    # 文本编码
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128)
    # 将编码数据转换为PyTorch张量
    input_ids = torch.tensor(encoding['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0).to(device)
    # 执行前向传播，得到logits
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
    # 预测结果
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

if __name__ == '__main__':
    # train(epochs=10)
    # 加载模型
    model.load_state_dict(torch.load('./bert_model.pth'))
    label1 = predict("味道不错，送的也很快！")
    label2 = predict("味道很一般，份量也不够！")
    print(label1, label2)  # 输出 1, 0 结果正确


