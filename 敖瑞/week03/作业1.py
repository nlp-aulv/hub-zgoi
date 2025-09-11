import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset


def get_dataset(file):
    dataset = pd.read_csv(file, sep='\t', header=None)
    texts = dataset[0].tolist()
    string_labels = dataset[1].tolist()

    label_to_index = {label: i for i, label in enumerate(set(string_labels))}
    index_to_label = {i: label for label, i in label_to_index.items()}
    numerical_labels = [label_to_index[label] for label in string_labels]

    char_to_index = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)

    index_to_char = {i: char for char, i in char_to_index.items()}
    vocab_size = len(char_to_index)
    return texts, string_labels, numerical_labels, label_to_index, index_to_label, char_to_index, index_to_char, vocab_size


class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[item]


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        # num_layers, batch_size, hidden_dim
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # batch_size, seq_len  -->  batch_size, seq_len, embedding_dim
        embedded = self.embedding(x)
        # batch_size, seq_len, embedding_dim  -->  batch_size, seq_len, hidden_dim
        out, hn = self.gru(embedded, h0)
        # batch_size, seq_len, hidden_dim -> batch_size, hidden_dim -> batch_size, output_dim
        out = self.layer(out[:, -1, :])
        return out


def train(texts, numerical_labels, char_to_index, max_len, embedding_dim, hidden_dim, num_layers, output_dim, vocab_size):
    gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
    dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

    model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        for idx, (inputs, outputs) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, outputs)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # if idx % 50 == 0:
            #     print('Batch: %d, Loss: %f' % (idx, loss.item()))
        print('Epoch: %d, Avg Loss: %f' % (epoch + 1, np.mean(train_loss)))

    return model


def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred = model(input_tensor)

    _, predicted_index = torch.max(pred, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]
    return predicted_label


if __name__ == '__main__':
    texts, string_labels, numerical_labels, label_to_index, index_to_label, char_to_index, \
    index_to_char, vocab_size = get_dataset('../week01/dataset.csv')
    max_len = 50
    embedding_dim= 64
    hidden_dim = 128
    num_layers = 2
    output_dim = len(label_to_index)
    model = train(texts, numerical_labels, char_to_index, max_len, embedding_dim, hidden_dim, num_layers, output_dim, vocab_size)
    new_text = '导航到萧山机场'
    pred = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
    print('输入为：%s，预测为：%s' % (new_text, pred))
