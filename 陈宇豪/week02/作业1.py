import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 索引化，人工进行向量embedding
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 为每个标签找到对应的索引向量
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activate):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activate = activate
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activate(out)
        out = self.fc2(out)
        return out


class SimpleClassifier2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activate):
        super(SimpleClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activate = activate
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.activate2 = activate
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activate(out)
        out = self.fc2(out)
        out = self.activate2(out)
        out = self.fc3(out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=50, shuffle=True)

hidden_dim = 128
output_dim = len(label_to_index)
models = [SimpleClassifier(vocab_size, hidden_dim, output_dim, nn.ReLU()),
          SimpleClassifier(vocab_size, hidden_dim, output_dim, nn.Sigmoid()),
          SimpleClassifier2(vocab_size, hidden_dim, output_dim, nn.ReLU()),
          SimpleClassifier2(vocab_size, hidden_dim, output_dim, nn.Sigmoid())]
criterion = nn.CrossEntropyLoss()
optimizers = [optim.Adam(models[0].parameters(), lr=0.01), optim.Adam(models[1].parameters(), lr=0.01),
              optim.Adam(models[2].parameters(), lr=0.01), optim.Adam(models[3].parameters(), lr=0.01)]

num_epochs = 10
epoch_loss = [[], [], [], []]
batch_loss = [[], [], [], []]
for epoch in range(num_epochs):
    for model in models:
        model.train()
    running_loss = [0.0, 0.0, 0.0, 0.0]
    for idx, (inputs, labels) in enumerate(dataloader):
        for i, model in enumerate(models):
            # 梯度清零
            optimizers[i].zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 损失计算
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizers[i].step()
            batch_loss[i].append(loss.item())
            running_loss[i] += loss.item()
    for i, loss in enumerate(running_loss):
        epoch_loss[i].append(np.round(loss / len(dataloader), 4))
    print(f"relu Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss[0] / len(dataloader):.4f}")
    print(f"sigmod Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss[1] / len(dataloader):.4f}")
    print(f"relu2 Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss[2] / len(dataloader):.4f}")
    print(f"sigmod2 Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss[3] / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)
    # 切换为预测模式
    model.eval()
    # 不计算梯度
    with torch.no_grad():
        output = model(bow_vector)
    # 选择最大的类别
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 损失函数绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))
x_label = ["relu激活函数一层隐藏层模型", "sigmod激活函数一层隐藏层模型",
           "relu激活函数两层隐藏层模型", "sigmod激活函数两层隐藏层模型"]
# 图1: Epoch损失对比
for i, (losses, name, color) in enumerate(zip(epoch_loss, x_label, ['r', 'g', "b", "y"])):
    epochs = range(1, len(losses) + 1)
    ax1.plot(epochs, losses, 'o-', linewidth=3, markersize=6,
             label=name, color=color, alpha=0.8)

    # 添加数值标签（每隔几个epoch显示一次）
    step = max(1, len(losses) // 10)
    for j in range(0, len(losses), step):
        ax1.annotate(f'{losses[j]:.3f}',
                     xy=(j + 1, losses[j]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, alpha=0.7)

ax1.set_xlabel('训练轮次 (Epoch)', fontsize=14, fontweight='bold')
ax1.set_ylabel('损失值', fontsize=14, fontweight='bold')
ax1.set_title('Epoch损失对比', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# 图2: Batch损失对比
for i, (losses, name, color) in enumerate(zip(batch_loss, x_label, ['r', 'g', "b", "y"])):
    batches = range(1, len(losses) + 1)
    ax2.plot(batches, losses, '-', linewidth=2,
             label=name, color=color, alpha=0.7)

ax2.set_xlabel('批次 (Batch)', fontsize=14, fontweight='bold')
ax2.set_ylabel('损失值', fontsize=14, fontweight='bold')
ax2.set_title('Batch损失对比', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
new_text_2 = "查询明天北京的天气"
for i, model in enumerate(models):
    predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"模型{i} 输入 '{new_text}' 预测为: '{predicted_class}'")
    predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"模型{i} 输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
