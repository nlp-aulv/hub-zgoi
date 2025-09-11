import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据加载和预处理保持不变
dataset = pd.read_csv("D:\AI\week\Week01\Week01\dataset.csv", sep="\t", header=None)
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


# 修改后的模型类，支持自定义层数和每层节点数
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FlexibleClassifier, self).__init__()
        layers = []
        prev_dim = input_dim

        # 动态创建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 准备数据
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义不同的模型配置
model_configs = [
    {"name": "原始模型(128)", "hidden_dims": [128]},
    {"name": "更宽网络(256)", "hidden_dims": [256]},
    {"name": "更窄网络(64)", "hidden_dims": [64]},
    {"name": "两层网络(128-64)", "hidden_dims": [128, 64]},
    {"name": "三层网络(256-128-64)", "hidden_dims": [256, 128, 64]},
    {"name": "深层网络(512-256-128-64)", "hidden_dims": [512, 256, 128, 64]}
]

output_dim = len(label_to_index)
num_epochs = 10

# 存储每个模型的loss历史
loss_history = {}

for config in model_configs:
    print(f"\n训练模型: {config['name']}")

    # 初始化模型
    model = FlexibleClassifier(vocab_size, config['hidden_dims'], output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    config_losses = []
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

        epoch_loss = running_loss / len(dataloader)
        config_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    loss_history[config['name']] = config_losses

# 打印所有模型的loss变化对比
print("\n===== 模型Loss对比 =====")
for config_name, losses in loss_history.items():
    print(f"{config_name}: 初始Loss {losses[0]:.4f} -> 最终Loss {losses[-1]:.4f}")

# 选择最佳模型进行预测（这里选择最终loss最小的模型）
best_model_name = min(loss_history, key=lambda k: loss_history[k][-1])
print(f"\n选择最佳模型: {best_model_name}")

# 重新训练最佳模型用于预测
best_config = next(c for c in model_configs if c['name'] == best_model_name)
model = FlexibleClassifier(vocab_size, best_config['hidden_dims'], output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练最佳模型
for epoch in range(num_epochs):
    model.train()
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# 预测函数保持不变
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


index_to_label = {i: label for label, i in label_to_index.items()}

# 使用最佳模型进行预测
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
