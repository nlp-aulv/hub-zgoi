# 理解rnn、lstm、gru的计算过程（⾯试⽤途），阅读官⽅⽂档 ：https://docs.pytorch.org/docs/2.4/nn.html#r
#  ecurrent-layers 最终 使⽤ GRU 代替 LSTM 实现05_LSTM⽂本分类.py

# RNN的计算过程：
"""
RNN是基础的循环神经网络，使用一个隐藏层来捕获历史信息，在处理某个具体的神经单元的时候，RNN将上⼀个时间步的隐藏状态和当前的输入作为输入，并生成新的隐藏和输出。
RNN模型在处理问题时候容易产生长效时间错误，他认为时间越久和当前的神经元月没有关系，导致进入梯度消失或者梯度爆炸，
RNN的计算过程：
第一步、定义所需变量
设置输入input_size=1、输出hide_size=10、和sequence_size=1,
第二步、初始化模型
rnn = nn.RNN(input_size,hide_size,batch_first=True)
第三步、准备输入数据
x = torch.randn(1,sequence_size,input_size)
第四步、初始化隐藏状态
h0 = torch.zeros(1, 1, hidden_size)
第五步、前向传播，输出结果，上一个输出是下一个的输入，隐藏层在初始化里面就设定了
output, hn = rnn(x, h0)
print(output)
print(hn)
如果是手动计算的话则使用一个正弦函数
"""
#lstm的计算过程：
"""
LSTM是RNN的改进版，是通过一个叫做门的机制来处理RNN所产生的梯度消失或者梯度爆炸的问题，这个门分为输入门，输出门，遗忘门，这三个门相互配合能精确处理数据在隐藏层的流动，从而解决长效距离依赖关系，
让模型更加准确。

第一步：定义所需变量
input_size=1,hidden_size=10,sequence_length=1
第二步：初始化模型
lstm = nn.LSTM(input_size,hidden_size,batch_first=True )

第三步：初始化数据：
x= torch.randn(1, sequence_length, input_size)

第四步：初始化隐藏状态和神经元每一个细胞状态，这里就是设定了两个状态h0,c0，手动实现的话需要定义输出门，输出门，遗忘门。
h0 = torch.zeros(1, 1, hidden_size)
c0 = torch.zeros(1, 1, hidden_size)

第五步输出：前向传播
output,(cn,hn) = lstm(x,(h0,c0))



"""
#gru的计算过程：
"""
GRU也是使用门的机制来解决RNN的问题，但是他更加简单，只有更新门和重置门，参数小，效率高，性能和LSTM相当。
具体计算过程：
第一步：一、定义变量输入输出和隐藏层，
input_size=1,hidden_size=10,sequence_length=1
第二步：定义模型
gru = nn.GRU(input_size,hidden_size,batch_first=True)
第三步：定义初始化数据
x= torch.randn(1, sequence_length, input_size)
第四步：定义初始化隐藏层数据
h0 = torch.zeros(1, 1, hidden_size)

第五步：前向传播
output,hn = gru(x,h0)
比lstm少了一输出，cn,
"""

# 具体代码
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
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

class CharGruDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- NEW LSTM Model Class ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hn = self.gru(embedded)
        out = self.fc(hn.squeeze(0))
        return out

# --- Training and Prediction ---
gru_dataset = CharGruDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
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
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "计分析今年程序员失业率给出合理建议"
predicted_class = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "分析最近A股行情"
predicted_class_2 = classify_text_gru(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
