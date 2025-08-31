import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# 自定义数据集类
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为张量
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()  # 创建词袋向量

    def _create_bow_vectors(self):
        # 将文本转换为索引序列
        tokenized_texts = []
        for text in self.texts:
            # 将字符转换为索引，截断或填充到max_len长度
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))  # 填充
            tokenized_texts.append(tokenized)

        # 创建词袋向量
        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)  # 初始化全零向量
            # 统计每个字符出现的次数
            for index in text_indices:
                if index != 0:  # 忽略填充字符
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)  # 将所有向量堆叠成张量

    def __len__(self):
        return len(self.texts)  # 返回数据集大小

    def __getitem__(self, idx):
        # 返回指定索引的数据样本
        return self.bow_vectors[idx], self.labels[idx]


# 定义不同结构的神经网络分类器
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleClassifier, self).__init__()
        self.layers = nn.ModuleList()

        # 创建多个隐藏层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集，使用制表符作为分隔符，没有表头
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()  # 获取文本列并转换为列表
string_labels = dataset[1].tolist()  # 获取标签列并转换为列表

# 将字符串标签映射为数字标签
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符到索引的映射表，用于将字符转换为数字
char_to_index = {'<pad>': 0}  # 填充字符的索引为0
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 创建索引到字符的反向映射表
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)  # 词汇表大小

max_len = 40  # 文本最大长度，超过部分截断，不足部分填充

# 创建数据集实例
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)

# 定义不同的模型配置
model_configs = [
    {"name": "(1层, 64节点)", "hidden_dims": [64], "lr": 0.01},
    {"name": "(1层, 128节点)", "hidden_dims": [128], "lr": 0.01},
    {"name": "(1层, 256节点)", "hidden_dims": [256], "lr": 0.01},
    {"name": "(2层, 64节点)", "hidden_dims": [64, 64], "lr": 0.01},
    {"name": "(2层, 128节点)", "hidden_dims": [128, 128], "lr": 0.01},
    {"name": "(2层, 256节点)", "hidden_dims": [256, 256], "lr": 0.01},
    {"name": "(3层, 128节点)", "hidden_dims": [128, 128, 128], "lr": 0.01},
]

# 存储每个模型的训练结果
results = {}

# 训练不同配置的模型
num_epochs = 15
output_dim = len(label_to_index)

for config in model_configs:
    print(f"\n训练模型: {config['name']}")
    print("=" * 50)

    # 创建数据加载器
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

    # 创建模型实例
    model = SimpleClassifier(vocab_size, config["hidden_dims"], output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    # 存储每个epoch的loss
    epoch_losses = []

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
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 存储结果
    results[config["name"]] = epoch_losses

# 绘制loss曲线对比图
plt.figure(figsize=(12, 8))
for model_name, losses in results.items():
    plt.plot(range(1, num_epochs + 1), losses, label=model_name, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型结构的Loss变化对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印最终loss对比
print("\n最终Loss对比:")
print("-" * 40)
for model_name, losses in results.items():
    print(f"{model_name}: {losses[-1]:.4f}")

# 选择最佳模型进行测试
best_model_name = min(results, key=lambda x: results[x][-1])
print(f"\n选择最佳模型进行测试: {best_model_name}")

# 重新训练最佳模型
best_config = next(config for config in model_configs if config["name"] == best_model_name)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
best_model = SimpleClassifier(vocab_size, best_config["hidden_dims"], output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(best_model.parameters(), lr=best_config["lr"])

# 训练最佳模型
for epoch in range(num_epochs):
    best_model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = best_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"最佳模型训练 - Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 定义文本分类函数
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将文本转换为索引序列
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))  # 填充

    # 创建词袋向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:  # 忽略填充字符
            bow_vector[index] += 1

    # 添加批次维度
    bow_vector = bow_vector.unsqueeze((0))

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        output = model(bow_vector)  # 前向传播

    # 获取预测结果
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]  # 将索引转换回标签

    return predicted_label


# 创建索引到标签的映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试最佳模型
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放周杰伦的音乐",
    "打开相机应用",
    "设置明天早上七点的闹钟"
]

print(f"\n使用最佳模型 '{best_model_name}' 进行测试:")
print("=" * 50)
for text in test_texts:
    predicted_class = classify_text(text, best_model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入 '{text}' 预测为: '{predicted_class}'")