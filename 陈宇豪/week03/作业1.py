import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
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


class CharLSTMDataset(Dataset):
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
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, is_use_gru=True, is_use_lstm=True):
        super(LSTMClassifier, self).__init__()
        self.is_use_gru = is_use_gru
        self.is_use_lstm = is_use_lstm
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 全链接层
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)

        # LSTM 层
        input_dim = embedding_dim
        if is_use_lstm:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            input_dim = hidden_dim

        # GRU层
        if is_use_gru:
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.embedding(x)
        if self.is_use_lstm:
            lstm_out, (hidden, cell_state) = self.lstm(out)
            # 取最后一个时间步或最后一层的隐藏状态
            out = hidden[-1]  # [batch_size, hidden_dim]

        if self.is_use_gru:
            if self.is_use_lstm:
                gru_out, hidden = self.gru(lstm_out)
            else:
                gru_out, hidden = self.gru(out)
            # 取最后一层的隐藏状态
            out = hidden[-1]  # [batch_size, hidden_dim]
        if not self.is_use_gru and not self.is_use_lstm:
            # 对序列维度取平均
            out = torch.mean(out, dim=1)  # [batch_size, embedding_dim]
            out = self.fc1(out)  # [batch_size, hidden_dim]
        out = self.fc(out)
        return out


# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

is_use_lstm = False
is_use_gru = False
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, is_use_lstm=is_use_lstm,
                       is_use_gru=is_use_gru)
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

torch.save(model.state_dict(), "classifier-lstm:{}-gru:{}.pth".format(is_use_lstm, is_use_gru))


def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
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

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

while True:
    user_input = input("请输入文本：")
    if user_input.lower() == "exit":
        break
    predicted_class = classify_text_lstm(user_input, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{user_input}' 预测为: '{predicted_class}'")
