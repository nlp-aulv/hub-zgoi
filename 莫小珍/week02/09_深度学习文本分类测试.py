import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# ... (Data loading and preprocessing remains the same) ...
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


# 新的可配置模型类
class ConfigurableClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        """
        Args:
            input_dim: 输入维度 (vocab_size)
            hidden_dims: 列表，包含各隐藏层的节点数，如 [128], [256, 128], [512, 256, 128]
            output_dim: 输出维度 (类别数)
            dropout_rate: Dropout比率
        """
        super(ConfigurableClassifier, self).__init__()

        layers = []
        current_dim = input_dim

        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 分割训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels
)

# 创建训练和验证数据集
train_dataset = CharBoWDataset(train_texts, train_labels, char_to_index, max_len, vocab_size)
val_dataset = CharBoWDataset(val_texts, val_labels, char_to_index, max_len, vocab_size)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定义不同的模型配置
model_configs = {
    "1层-64节点": {"hidden_dims": [64]},
    "1层-128节点": {"hidden_dims": [128]},
    "1层-256节点": {"hidden_dims": [256]},
    "2层-128-64": {"hidden_dims": [128, 64]},
    "2层-256-128": {"hidden_dims": [256, 128]},
    "3层-256-128-64": {"hidden_dims": [256, 128, 64]},
    "3层-512-256-128": {"hidden_dims": [512, 256, 128]},
    "4层-256-128-64-32": {"hidden_dims": [256, 128, 64, 32]}
}

# 存储训练结果
results = {}
output_dim = len(label_to_index)


# 训练函数
def train_model(model_config, config_name, num_epochs=15):
    print(f"\n=== 训练配置: {config_name} ===")

    model = ConfigurableClassifier(
        input_dim=vocab_size,
        hidden_dims=model_config["hidden_dims"],
        output_dim=output_dim,
        dropout_rate=0.2
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_epoch_loss = train_loss / len(train_loader)
        val_epoch_loss = val_loss / len(val_loader)
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'model': model,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }


# 训练所有配置
for config_name, config in model_configs.items():
    results[config_name] = train_model(config, config_name, num_epochs=15)

# 可视化对比结果
plt.figure(figsize=(14, 10))

# 绘制训练loss
plt.subplot(2, 1, 1)
colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
for i, (config_name, result) in enumerate(results.items()):
    plt.plot(result['train_loss'], label=config_name, color=colors[i], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('不同模型结构的训练Loss对比')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 绘制验证loss
plt.subplot(2, 1, 2)
for i, (config_name, result) in enumerate(results.items()):
    plt.plot(result['val_loss'], label=config_name, color=colors[i], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('不同模型结构的验证Loss对比')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 创建对比表格
print("\n=== 最终性能对比 ===")
comparison_data = []
for config_name, result in results.items():
    comparison_data.append({
        '模型配置': config_name,
        '层数': len(model_configs[config_name]["hidden_dims"]),
        '总参数数': sum(p.numel() for p in result['model'].parameters()),
        '最终训练Loss': f"{result['final_train_loss']:.4f}",
        '最终验证Loss': f"{result['final_val_loss']:.4f}",
        '过拟合程度': f"{(result['final_train_loss'] - result['final_val_loss']):.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.sort_values('最终验证Loss'))

# 选择最佳模型进行预测
best_config_name = min(results.items(), key=lambda x: x[1]['final_val_loss'])[0]
best_model = results[best_config_name]['model']

print(f"\n最佳模型配置: {best_config_name}")
print(f"最终验证Loss: {results[best_config_name]['final_val_loss']:.4f}")


# 预测函数
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
predicted_class = classify_text(new_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")