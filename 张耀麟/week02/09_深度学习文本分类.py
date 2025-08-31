from collections import OrderedDict
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from SimpleClassifier import SimpleClassifier


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


# 单句子估计
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
    predicted_label = index_to_label[predicted_index.item()]

    return predicted_label


# 模型训练
def train_model(num_epochs, model: SimpleClassifier, train_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #     if idx % 50 == 0:
        #         print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}")

    model.eval()
    testing_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            testing_loss += loss.item()
    print('=' * 20)
    print(f"Model layer size: [{model.layers_size()}] \n"
          f"Model shape : [{model.nodes_of_layers()}] \n"
          f"Average loss in dataset = [{testing_loss}]")
    print('=' * 20)


if __name__ == '__main__':
    # ... (Data loading and preprocessing remains the same) ...
    dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
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
    max_len = 40

    char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
    train_loader = DataLoader(char_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(char_dataset, batch_size=len(char_dataset))

    model1 = SimpleClassifier(input_dims=[vocab_size, 128], output_dims=[128, len(label_to_index)])
    model2 = SimpleClassifier(input_dims=[vocab_size, 64, 128], output_dims=[64, 128, len(label_to_index)])
    model3 = SimpleClassifier(input_dims=[vocab_size, 128, 128], output_dims=[128, 128, len(label_to_index)])
    model4 = SimpleClassifier(input_dims=[vocab_size, 64, 64], output_dims=[64, 64, len(label_to_index)])

    num_epochs = 10
    train_model(num_epochs, model1, train_loader, test_loader)
    train_model(num_epochs, model2, train_loader, test_loader)
    train_model(num_epochs, model3, train_loader, test_loader)
    train_model(num_epochs, model4, train_loader, test_loader)

# 单句预测
# new_text = "帮我导航到北京"
# predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
# print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
#
# new_text_2 = "查询明天北京的天气"
# predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
# print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
