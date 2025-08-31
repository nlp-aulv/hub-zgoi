import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class TextDataset(Dataset):
    def __init__(self, textList, text_labels, char_to_index, vector_size):
        self.textList = textList
        self.char_to_index = char_to_index
        self.text_labels = text_labels
        self.vector_size = vector_size
        self.max_length = 40
        self.low_vectors = self._create_bow_vectors()

    def   _create_bow_vectors(self):
        tokenize_texts = []
        for text in self.textList:
            temp = [self.char_to_index[word] for word in text[:self.max_length]]
            temp += [0] * (self.max_length - len(temp))
            tokenize_texts.append(temp)
        low_vectors = []
        for tokenize_text in tokenize_texts:
            low_vector = torch.zeros(self.vector_size)
            for token in tokenize_text:
                if token >= 2823:
                    print("findit")
                if token != 0:
                    low_vector[token] += 1
                low_vector[token] += 1
            low_vectors.append(low_vector)
        return torch.stack(low_vectors)

    def __getitem__(self, item):
        x = self.low_vectors[item]
        y = self.text_labels[item]
        return x, y
    def __len__(self):
        return len(self.textList)

class ClassificicatModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(ClassificicatModel,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        # 添加批归一化
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        y = self.fc2(x)
        return y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据集构建
dateset = pd.read_csv('dataset.csv',header=None,sep="\t", )
textList = dateset[0].tolist()
labelList = dateset[1].tolist()
#获取字符，类别映射
label_to_index = { name:index for index,name  in enumerate(set(labelList))}
index_to_label = { index:name for index,name  in enumerate(set(labelList))}
char_to_index = {'<pan>':0}
for text in textList:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

text_labels = [label_to_index[classes] for classes in labelList]
vector_size = len(char_to_index)
max_length = 40

train_text,test_text, train_label,test_label = train_test_split(textList, text_labels,test_size=0.2)

trainDataset = TextDataset(train_text, train_label, char_to_index, vector_size)
textDataset = TextDataset(test_text,test_label, char_to_index,vector_size)
train_dataloader = torch.utils.data.DataLoader(trainDataset,batch_size=32,shuffle=True)
val_dataloader = torch.utils.data.DataLoader(textDataset,batch_size=32,shuffle=True)

#模型训练-不同神经元的模型
for i in range(3):
    model = ClassificicatModel(len(char_to_index),100*(i+1),len(index_to_label)).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_losses = []
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for idx,(inputs,labels) in enumerate(train_dataloader):
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            total_train += labels.size(0)
            a, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            # if idx % 50 == 0:
            #     print(f"Epoch {epoch} Loss {loss.item():.4f}")
        total_acc = 100*correct_train / total_train
        total_loss = running_loss / len(train_dataloader)
        train_accuracies.append(total_acc)
        train_losses.append(total_loss)
        # print(f"Epoch {epoch} Loss {total_loss :.4f}")

        model.eval()

        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                val_loss += loss.item()

                # 计算验证准确率
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            val_accuracy = 100 * correct_val / total_val
            val_loss = val_loss / len(val_dataloader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f"Exporch:{epoch} Accuracy {val_accuracy :.4f}")



    #训练过程中训练集与测试机准确率的验证
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy',)
    plt.plot(val_accuracies, label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


