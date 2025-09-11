import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据加载和预处理
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
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


# 数据集类定义
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


# 调整后的神经网络模型（增加层数和节点数）
class EnhancedClassifier(nn.Module):
    # 输入层维度，隐藏层维度，输出层维度
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(EnhancedClassifier, self).__init__()

        # 创建层列表
        layers = []
        in_features = input_dim

        # 添加隐藏层（根据hidden_dims中的维度）
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())  # 添加ReLU激活函数
            in_features = hidden_dim  # 更新下一层的输入维度

        # 添加输出层
        layers.append(nn.Linear(in_features, output_dim))

        # 组合所有层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 创建数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 模型配置 - 调整层数和节点数
input_dim = vocab_size
output_dim = len(label_to_index)

# 单隐藏层128节点 → 调整后：三层隐藏层（256, 512, 256节点）
model = EnhancedClassifier(input_dim, [256, 512, 256], output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 改用Adam优化器

# 训练
num_epochs = 15  # 增加训练轮次
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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{idx}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)  # 添加batch维度

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 创建索引到标签的映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试模型
test_queries = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放周杰伦的歌曲",
    "设置明天早上7点的闹钟",
    "今天的股票行情怎么样"
]

for query in test_queries:
    predicted_class = classify_text(query, model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入 '{query}' 预测为: '{predicted_class}'")