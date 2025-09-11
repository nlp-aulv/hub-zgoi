import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 加载数据集，并进行预处理
dataset = pd.read_csv("../week1/dataset.csv", sep="\t", header=None)
# 获取文本数据并转换为列表
texts = dataset[0].tolist()
# 获取标签并转换为列表
string_labels = dataset[1].tolist()

# 创建标签到索引的映射字典
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将字符串标签转换为数值标签
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符到索引的映射字段，初始包含填充符
char_to_index = {'<pad>': 0}
# 遍历所有文本，构建字符到索引的完整映射
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 创建索引到字符的反向映射
index_to_char = {i: char for char, i in char_to_index.items()}
# 获取词汇表大小
vocab_size = len(char_to_index)

# 设置文本的最大长度
max_len = 40


# 自定义数据集类，处理字符级别的Bag-of-Wolds表示
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        # 将标签转换为PyTorch张量 -- 接收标签，然后转换为张量；指定张量类型为长整型(64位整熟) 等价与torch.int64
        # PyTorch 模型和函数需要张量作为输出，而不是普通的python列表
        # 张量可以轻松的已转到GPU上进行加速计算
        # 张量可以高效的处理批量数据
        # 自动梯度计算：标签通常不需要梯度，但保持一致性很重要
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()  # 创建Bag-of-Worlds向量

    def _create_bow_vectors(self):
        tokenized_texts = []
        # 将文本转换为索引序列，并进行填充/截断
        for text in self.texts:
            # 获取每个字符的索引，未知字符使用0填充
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 填充到最大长度
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        # 为每个文本创建Bag-of-World 向量
        for text_indices in tokenized_texts:
            # 初始化全零向量
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                # 忽略填充符
                if index != 0:
                    # 对应字符计数+1
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        # 将列表中的向量堆叠成张量
        return torch.stack(bow_vectors)

    def __len__(self):
        # 返回数据集大小
        return len(self.texts)

    def __getitem__(self, idx):
        # 返回指定索引的样本
        return self.bow_vectors[idx], self.labels[idx]


# 定义简单的神经网络分类器
class SimpleClassifier(nn.Module):
    # def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
    # super(SimpleClassifier, self).__init__()
    # 输入层到隐藏层的全连接层
    # self.fc1 = nn.Linear(input_dim, hidden_dim)
    # ReLU激活函数
    # self.relu = nn.ReLU()
    # 隐藏层到输出层的全链接层
    # self.fc2 = nn.Linear(hidden_dim, output_dim)

    # def forward(self, x):
    #     # 手动实现每层的计算
    #     # 第一层线性变换
    #     out = self.fc1(x)
    #     # 应用ReLU激活函数
    #     out = self.relu(out)
    #     # 第二层线性变换
    #     out = self.fc2(out)
    #     # 返回输出，未应用softmax，因为CrossEntropyLoss内部会处理
    #     return out

    def __init__(self, input_dims):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        # 使用nn.ModuleList 动态创建层
        self.inputs = nn.ModuleList()
        for i in range(len(input_dims) - 1):
            self.inputs.append(nn.Linear(input_dims[i], input_dims[i + 1]))

        self.relu = nn.ReLU()

    def forward(self, x):
        # 遍历所有层
        for item in self.inputs[:-1]:
            x = self.relu(item(x))

        # 最后一层不使用激活函数
        x = self.inputs[-1](x)
        return x

# 创建数据集实列
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
# 创建数据加载器，用于批量加载数据
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)  # 读取批量数据集 -》 batch数据

# 定义模型参数
# 隐藏层维度
hidden_dim = 128
# 输出维度等于类别数量
output_dim = len(label_to_index)
# 初始化模型
# model = SimpleClassifier(vocab_size, hidden_dim, output_dim)  # 维度和精度有什么关系？
model = SimpleClassifier([vocab_size, 512, 256, 128, 64, output_dim])
criterion = nn.CrossEntropyLoss()  # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.5)  # 定义优化器 随机梯度下降

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

# 训练模型
# 训练轮数
num_epochs = 10
for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
    # 设置模型为训练模式
    model.train()
    # 累计损失
    running_loss = 0.0
    # 遍历数据加载器中的每个批次
    for idx, (inputs, labels) in enumerate(dataloader):
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失值
        running_loss += loss.item()
        # 每50个批次打印一次损失
        if idx % 100 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 打印每个epoch的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 定义文本分类函数
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将输入文本转换为索引序列
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 填充到最大长度
    tokenized += [0] * (max_len - len(tokenized))

    # 创建Big-of-Worlds向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    # 添加批次维度
    bow_vector = bow_vector.unsqueeze(0)

    # 设置模型为评估模型
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 向前传播
        output = model(bow_vector)

    # 获取预测结果
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    # 将索引转换回标签
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 创建索引到标签的映射
index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "写一段Python代码"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
