import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ...数据加载...
dataset = pd.read_csv("dataset.csv", sep='\t', header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# ... 对标签进行编码 ...
label_to_index = {label: i for i, label in enumerate(set(string_labels))}  # 对字符串标签去重并映射为编码字典(标签为键)
numerical_labels = [label_to_index[label] for label in string_labels]  # 将文本对应的字符串标签编码

# ... 对词（字）进行编码 ...
char_to_index = {'<pad>': 0}  # 初始化字典 字:编码
for text in texts:  # 对每一个文本遍历
    for char in text:  # 对每一个字遍历
        if char not in char_to_index:  # 如果该字不在这个字典里
            char_to_index[char] = len(char_to_index)  # 添加到该字典

index_to_char = {i: char for char, i in char_to_index.items()}  # 将字典键值对调转化为 编码:字
vocab_size = len(char_to_index)  # 计算字典长度

max_len = 40  # 设置文本长度


# 处理文本
class CharBoWDataset(Dataset):  # 继承Dataset的类
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):  # 定义构造方法
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将传入的标签转为tensor类型
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vector = self._create_bow_vectors()  # 创建文本词频

    def _create_bow_vectors(self):  # 创建文本词频
        tokenized_texts = []  # 初始化 文本库
        for text in self.texts:  # 将文本里的字转为编码
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]  # 处理文本里的字，在字典里找到对应的编码
            tokenized += [0] * (self.max_len - len(tokenized))  # 不足长度的补足
            tokenized_texts.append(tokenized)  # 加入 文本库 里
        bow_vectors = []  # 初始化
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)  # 初始化
            for index in text_indices:
                    if index != 0:
                        bow_vector[index] += 1  # 计算文本内对应的词频
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    # 对DataLoader的接口1
    def __len__(self):  # 返回样本个数
        return len(self.texts)

    # 对DataLoader的接口2
    def __getitem__(self, idx):  # 返回 第idx个 文本 对应 的词频和编码标签
        return self.bow_vector[idx], self.labels[idx]


# 建立模型
class SimpleClassifier(nn.Module):  # 继承nn.Module的类
    def __init__(self, input_dim, hidden_dims, output_dim):  # 多层
        super(SimpleClassifier, self).__init__()

        layers = []  # 层列表初始化
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.5))
            input_dim = hidden_dim
        # 输出层
        layers.append(nn.Linear(input_dim, output_dim))
        # 将层列表转化
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out


# 多层模型训练
def model_train(model, dataloader, criterion, optimizer, num_epochs=10):
    all_epoch_loss = []  # 初始化 用于存储 Epoch 的平均批次损失
    for epoch in range(num_epochs):  # 每次一个数据集训练
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0  # 初始化，用于累计当前 Epoch 内所有批次的损失
        for idx, (inputs, labels) in enumerate(dataloader):  # 每次一个批次训练
            optimizer.zero_grad()  # 梯度清零，PyTorch 会累计梯度
            outputs = model(inputs)  # 前向传播，将输入数据 inputs 传入模型，经过所有层的计算，得到模型的预测输出 outputs
            loss = criterion(outputs, labels)  # 使用损失函数（这里是 CrossEntropyLoss）来计算模型的预测输出 outputs 与真实标签 labels 之间的差异
            loss.backward()  # 反向传播 自动计算损失函数相对于所有模型可学习参数的梯度（导数）
            optimizer.step()  # 更新模型参数
            running_loss += loss.item()  # 将当前批次的损失值累加到 running_loss 上
        all_epoch_loss.append(running_loss/len(dataloader))
    return model, all_epoch_loss  # 输出模型和批次平均loss


# epoch: 将数据集整体迭代训练一次
# batch: 数据集汇总为一批训练一次
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=50, shuffle=True)  # 批量读取数据集 -> batch数据

hidden_dims = [[64], [128], [256], [64, 64], [128, 128], [256, 256], [128, 64], [256, 128], [256, 128, 64]]  # 设置隐藏层
hidden_dims_tuple = [tuple(dim) for dim in hidden_dims]

output_dim = len(label_to_index)  # 设置输出层

models = {hidden_dim: None for hidden_dim in hidden_dims_tuple}
model_loss = {hidden_dim: None for hidden_dim in hidden_dims_tuple}

criterion = nn.CrossEntropyLoss()  # 损失函数 内部自带激活函数，softmax

for i, hidden_dim in enumerate(hidden_dims_tuple):
    model = SimpleClassifier(vocab_size, hidden_dim, output_dim)  # 模型
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器 可以结合梯度 动态调整学习
    model, epoch_loss = model_train(model, dataloader, criterion, optimizer, num_epochs=10)
    models[hidden_dim], model_loss[hidden_dim] = model, epoch_loss
    # print("模型为：", models[hidden_dim])
    # print("epoch批次平均loss=：", [f"{num:.4f}" for num in model_loss[hidden_dim]])
    # print("模型为：", model)
    print("模型为：", hidden_dim)
    print("epoch批次平均loss：", [f"{num:.4f}" for num in epoch_loss])


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]  # 将文本的字转为编码 最大长度max_len
    tokenized += [0] * (max_len - len(tokenized))  # 不足补齐0

    bow_vector = torch.zeros(vocab_size)  # 初始化一个torch，长度为字典长度
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)  # 增加一个维度

    model.eval()  # 评估模式
    with torch.no_grad():  # 使用 torch.no_grad() 上下文管理器禁用梯度计算
        outputs = model(bow_vector)  # 在推理阶段不需要计算梯度，这可以节省内存和计算资源

    _, predicted_index = torch.max(outputs, dim=1)  # _ 是最大值本身，找到输出向量中得分最高的类别的编码
    predicted_index = predicted_index.item()  # 提取值
    predicted_label = index_to_label[predicted_index]  # 将对应编码转换成字符串

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}  # 将字典键值对调转化为 编码:标签

new_texts = ["帮我导航到北京", "查询明天北京的天气"]  #

for new_text in new_texts:
    print(f"输入{new_text}")
    for hidden_dim, model in models.items():
        predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
        # print(f"模型：{model} 预测为：{predicted_class}")
        print(f"模型：{hidden_dim} 预测为：{predicted_class}")
