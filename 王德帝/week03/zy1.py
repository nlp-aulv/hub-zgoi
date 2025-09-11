import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import jieba
from sklearn.model_selection import train_test_split

'''
使用word2vec + gru进行文本分类
'''

# 1. 数据准备
data = pd.read_csv('./z_train/dataset.csv', encoding='utf-8', sep='\t' ,header=None)
texts = data[0].to_list()
labels_list = data[1].to_list()
lable_mapping = {lable : i for i, lable in enumerate(set(labels_list))}
index_to_lable = {i:lable for lable, i in lable_mapping.items()}

sentences = [jieba.lcut(text) for text in texts]

#划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(sentences, labels_list, test_size=0.2, random_state=66)

word2vec = Word2Vec(
    sentences = sentences, #list of list[str]
    vector_size = 100, #词向量维度
    window = 5,
    min_count = 1,
    sg = 1 # 1:skip-gram, 0:cbow
)

class textDataset(Dataset):
    def __init__(self, texts, labels, max_len ,word2vec):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.word2vec = word2vec

    def __len__(self): #返回数据集大小
        return len(self.texts)
    
    def __getitem__(self, idx):#返回第idx个数据
        text = self.texts[idx]
        #将文本转为向量
        vector = []
        for word in text:
            if word in self.word2vec.wv:
                vector.append(self.word2vec.wv[word])
            else:
                vector.append(torch.zeros(word2vec.vector_size))

        if len(vector) > self.max_len:
            vector = vector[:self.max_len]
        else:
            vector += [torch.zeros(word2vec.vector_size)] * (self.max_len - len(vector))
        
        vector = torch.tensor(vector,dtype=torch.float32)

        #标签转数字
        label = lable_mapping[self.labels[idx]]
        lable = torch.tensor(label,dtype=torch.long)
        return vector, lable

senquence_length = 50 #每个句子的最大长度,序列长度
batch_size = 32 #每个batch的大小

train_dataset = textDataset(train_texts, train_labels, senquence_length, word2vec) #(12100*0.8, 50 ,100)
test_dataset = textDataset(test_texts, test_labels, senquence_length, word2vec) #(12100*0.2, 50 ,100)

#(batch_size, senquence_length, embedding_size)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True) #每个批次: 文本向量(32 ,50 ,100) 标签向量(32)   
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False) #每个批次: 文本向量(32 ,50 ,100) 标签向量(32)

# # 获取第一批次数据并查看前两条数据的形状
# for vectors, labels in train_dataloader:
#     print(f"批次数据形状 - 文本向量: {vectors.shape}, 标签: {labels.shape}")
#     # 查看前两条数据的形状
#     for i in range(min(2, len(labels))):
#         print(vectors[i])
#         print(labels[i])
#     break

#2. 模型构建
class gruClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(gruClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x): #x: (batch_size, senquence_length, embedding_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0) #out: (batch_size, senquence_length, hidden_size)      
        out = self.fc(out[:, -1, :]) #out: (batch_size, output_size)
        return out
    
input_size = word2vec.vector_size #词向量的维度
hidden_size = 64 #隐藏层维度
output_size = len(lable_mapping) #输出层维度

model = gruClassifier(input_size, hidden_size, output_size)

#3. 模型训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0 #跟踪每个epoch的损失

    for vectors, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(vectors) #outputs: (batch_size, output_size)
        loss = criterion(outputs, labels) #计算损失
        loss.backward() #反向传播
        optimizer.step() #更新参数

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_dataloader) #计算每个epoch的平均损失

    model.eval()
    total = 0
    correct = 0 
    with torch.no_grad(): #不需要计算梯度
        for vectors, labels in test_dataloader:
            outputs = model(vectors)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}/{epochs},Loss: {epoch_loss:.4f}, Accuracy: {100 * correct / total:.2f}%')

#4.预测文本
def predict(text):
    #文本转张量
    text = jieba.lcut(text)
    vectors = []
    for word in text:
        if word in word2vec.wv:
            vectors.append(word2vec.wv[word])
        else:
            vectors.append(torch.zeros(word2vec.vector_size))
    if len(vectors) > senquence_length:
        vectors = vectors[:senquence_length]
    else:
        vectors = vectors +  [np.zeros(word2vec.vector_size)] * (senquence_length - len(vectors))
    
    print('vectors:',vectors)
    vectors = torch.tensor(vectors,dtype=torch.float32)  #(senquence_length,input_size)
    x = vectors.unsqueeze(0) #(batch_size, senquence_length, embedding_size)

    #正向传播
    model.eval()
    with torch.no_grad():
        out = model.forward(x)
    
    _,predicted_lable_index = torch.max(out,1)
    predicted_lable = index_to_lable[predicted_lable_index.item()]

    return predicted_lable

new_text = "帮我导航到北京"
result = predict(new_text)
print(f"输入 '{new_text}' 预测为: '{result}'")
