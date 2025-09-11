import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- 数据加载和预处理（与原代码相同）---
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 步骤1: 考虑输入、输出层
# 步骤2: 考虑中间的层，多少层？ 层神经元个数、激活函数
# 步骤3: 模型1效果 vs 模型2效果

char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
output_dim = len(label_to_index)

# --- 定义模型训练和评估函数 ---
def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(model, dataloader, num_epochs, criterion, optimizer):
    loss_history = []
    print(f"--- 训练模型，参数量: {get_parameter_count(model)} ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return loss_history

# --- 模型对比 ---
hidden_dim_1 = 64
hidden_dim_2 = 1024 # 模型复杂，需要更多训练时间，可能更加容易过拟合

# 模型 1：较小的隐藏层
model_small = SimpleClassifier(vocab_size, hidden_dim_1, output_dim)
optimizer_small = optim.Adam(model_small.parameters(), lr=0.05)

# 模型 2：较大的隐藏层
model_large = SimpleClassifier(vocab_size, hidden_dim_2, output_dim)
optimizer_large = optim.Adam(model_large.parameters(), lr=0.05)

criterion = nn.CrossEntropyLoss()
num_epochs = 20

# 训练并记录损失
loss_history_small = train_and_evaluate(model_small, dataloader, num_epochs, criterion, optimizer_small)
loss_history_large = train_and_evaluate(model_large, dataloader, num_epochs, criterion, optimizer_large)

# --- 绘制损失曲线 ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_history_small, label=f'Hidden Dim = {hidden_dim_1} (Parameters: {get_parameter_count(model_small)})')
plt.plot(range(1, num_epochs + 1), loss_history_large, label=f'Hidden Dim = {hidden_dim_2} (Parameters: {get_parameter_count(model_large)})')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# --- 预测部分（与原代码相同）---
# 为了演示，我们可以用其中一个模型进行预测
index_to_label = {i: label for label, i in label_to_index.items()}

def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1
    bow_vector = bow_vector.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]
    return predicted_label

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model_large, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model_large, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
