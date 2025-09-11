import pandas as pd
import numpy as np
import torch
from IPython.core.pylabtools import figsize
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, BertForSequenceClassification

from constants import *

# 读取数据
data = pd.read_csv(DATA_SET_LOCATION)
data["review"].astype(str)

# 查看信息
# print(data.info())
# print(data.head())
# print(f"缺失值数量:%s", data["review"].isnull().sum())
# print("review type:%s", data["review"].dtype)

# 拆分数据为训练集、测试集
train_data_pd = data.sample(frac=0.8, random_state=22, axis=0)
test_data_pd = data[~data.index.isin(train_data_pd.index)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 获取分词器
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)

# 对数据进行token化
# 固定长度，超过截断、不足padding
train_token_coding = tokenizer(train_data_pd["review"].values.tolist(), truncation=True, padding=True, max_length=64,
                               return_tensors="pt")
test_token_coding = tokenizer(test_data_pd["review"].values.tolist(), truncation=True, padding=True, max_length=64,
                              return_tensors="pt")


# 数据读取类
class DATAGET(Dataset):
    # 初始化数据集
    def __init__(self, encodings, labels):
        # 是一个字典类型 k-v
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    # 获取一个数据
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item


train_dataset = DATAGET(encodings=train_token_coding, labels=train_data_pd["label"])
test_dataset = DATAGET(encodings=test_token_coding, labels=test_data_pd["label"])

# 批量获取数据
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 读取预训练模型
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PERTRAINED_PATH, num_labels=12)
# 开始训练
model.to(device)
# 定义优化器，使用AdamW，lr是学习率
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)


# 计算准确度
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


train_loss = np.array([])
test_acc = np.array([])


def train(model):
    global train_loss
    model.train()
    for batch_idx, batch_data in enumerate(train_dataloader):
        # 清空梯度
        optim.zero_grad()
        input_data = batch_data['input_ids'].to(device)
        # 获取attention_mask
        attention_mask = batch_data['attention_mask'].to(device)
        # 获取标签
        labels = batch_data['labels'].to(device)
        # 获取输出
        outputs = model(input_data, attention_mask=attention_mask, labels=labels)
        # 获取loss
        loss = outputs[0]
        train_loss = np.append(train_loss, loss.cpu().detach().numpy())
        # 反向传播
        loss.backward()
        # 裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 更新参数
        optim.step()
        print("epoch: %d, iter_num: %d, loss: %.4f" % (epoch + 1, batch_idx + 1, loss.item()))


def evaluate(model):
    global test_acc
    model.eval()
    for batch_idx, batch_data in enumerate(test_dataloader):
        with torch.no_grad():
            # 清空梯度
            optim.zero_grad()
            input_data = batch_data['input_ids'].to(device)
            # 获取attention_mask
            attention_mask = batch_data['attention_mask'].to(device)
            # 获取标签
            labels = batch_data['labels'].to(device)
            outputs = model(input_data, attention_mask=attention_mask, labels=labels)

        logits = outputs[1]
        # 获取预测结果
        logits = logits.detach().cpu().numpy()
        # 获取真实标签
        labels = labels.to('cpu').numpy()
        # 计算准确率
        test_acc = np.append(test_acc, flat_accuracy(logits, labels))


if __name__ == '__main__':
    for epoch in range(1):
        print("------------Epoch: %d ----------------" % epoch)
        # 训练模型
        train(model)
        # 验证模型
        evaluate(model)

    # 保存模型
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    # 损失画图
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制训练损失
    axes[0].plot(train_loss, 'b-', linewidth=2)
    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # 绘制测试准确率
    axes[1].plot(test_acc, 'r-', linewidth=2)
    axes[1].set_title("Test Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()
    plt.show()
