import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# ========== 数据预处理部分保持不变 ==========
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

# ========== 模型定义 ==========
class SimpleClassifier(nn.Module):
    """单隐藏层"""
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

# class TwoHiddenClassifier(nn.Module):
#     """双隐藏层"""
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
#         super(TwoHiddenClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(hidden_dim2, output_dim)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         return out
class TwoHiddenClassifier(nn.Module):
    """单隐藏层,多神经元"""
    def __init__(self, input_dim, hidden_dim2, output_dim):
        super(TwoHiddenClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ========== 训练函数 ==========
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"[{model.__class__.__name__}] Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"模型 {model.__class__.__name__} 总训练时间: {total_time:.2f} 秒")
    return model

# ========== 数据 & 超参数 ==========
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

hidden_dim = 128
# hidden_dim2 = 128
hidden_dim2 = 1024
output_dim = len(label_to_index)

# ========== 训练单隐藏层模型 ==========
model1 = SimpleClassifier(vocab_size, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
train_model(model1, dataloader, criterion, optimizer1, num_epochs=10)

# ========== 训练双隐藏层模型 ==========
model2 = TwoHiddenClassifier(vocab_size, hidden_dim, hidden_dim2, output_dim)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)
train_model(model2, dataloader, criterion, optimizer2, num_epochs=10)
