import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset


# 转向量
class CharBowDataset(Dataset):
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

    def __getitem__(self, item):
        return self.bow_vectors[item], self.labels[item]


# 网络
class SimpleClassifier(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        layers = []
        for i in range(len(layer_size)-1):
            layers.append(nn.Linear(layer_size[i], layer_size[i+1]))

            if i < len(layer_size) - 2:
                layers.append(nn.ReLU())
        print('网络结构：', layers)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 获取数据集、词表、标签字典
def get_dataset(file):
    dataset = pd.read_csv(file, sep='\t', header=None)
    texts = dataset[0].tolist()
    string_labels = dataset[1].tolist()
    # 标签字典
    label_to_index = {label: i for i, label in enumerate(set(string_labels))}
    numerical_labels = [label_to_index[char] for char in string_labels]
    # 字典
    char_to_index = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)
    index_to_char = {i: char for char, i in char_to_index.items()}
    vocab_size = len(char_to_index)

    return texts, numerical_labels, char_to_index, index_to_char, label_to_index, vocab_size


# 训练
def train(layer_size, texts, numerical_labels, char_to_index, vocab_size, max_len):
    char_dataset = CharBowDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

    model = SimpleClassifier(layer_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epoch = 10
    for epoch in range(num_epoch):
        model.train()
        train_loss = []
        for index, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print('第 %d 轮，平均损失：%f' % (epoch+1, np.mean(train_loss)))
    return model


# 推理
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


if __name__ == '__main__':
    texts, numerical_labels, char_to_index, index_to_char, label_to_index, vocab_size = get_dataset('../week01/dataset.csv')
    max_len = 50
    index_to_label = {i: label for label, i in label_to_index.items()}
    layer_sizes = [[vocab_size, 64, len(label_to_index)],
                   [vocab_size, 128, 64, len(label_to_index)],
                   [vocab_size, 256, 32, len(label_to_index)],
                   [vocab_size, 256, 128, 64, len(label_to_index)]]
    for layer_size in layer_sizes:
        print('模型层数和节点数：', layer_size)
        model = train(layer_size, texts, numerical_labels, char_to_index, vocab_size, max_len)
        new_text = '杭州明天的天气怎么样'
        predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
        print(f'输入: {new_text}  预测为：{predicted_class}')
