import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)  # 生成-2π到2π之间的1000个点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # 生成带噪声的sin函数值

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("Sin函数数据生成完成。")
print(f"X范围: [{X.min().item():.2f}, {X.max().item():.2f}]")
print("---" * 10)


# 2. 定义多层神经网络
class SinNet(torch.nn.Module):
    def __init__(self, hidden_size=64, num_layers=3):
        super(SinNet, self).__init__()
        layers = []

        # 输入层
        layers.append(torch.nn.Linear(1, hidden_size))
        layers.append(torch.nn.ReLU())

        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.ReLU())

        # 输出层
        layers.append(torch.nn.Linear(hidden_size, 1))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 3. 创建模型、损失函数和优化器
model = SinNet(hidden_size=64, num_layers=4)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"模型结构: {model}")
print("---" * 10)

# 4. 训练模型
num_epochs = 2000
losses = []  # 记录损失变化

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失
    losses.append(loss.item())

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 6. 绘制拟合结果
plt.figure(figsize=(12, 6))

# 绘制原始数据和带噪声的数据
plt.scatter(X_numpy, y_numpy, label='Noisy sin data', color='blue', alpha=0.3, s=10)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin function', color='green', linewidth=2)

# 使用模型预测
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.plot(X_numpy, y_predicted, label='Neural network prediction', color='red', linewidth=2)

plt.title('Fitting sin function with neural network')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 7. 比较不同网络结构的效果
print("\n比较不同网络结构的效果...")

# 定义不同的网络结构
configs = [
    {"hidden_size": 16, "num_layers": 2, "label": "Small network (16, 2)"},
    {"hidden_size": 64, "num_layers": 3, "label": "Medium network (64, 3)"},
    {"hidden_size": 128, "num_layers": 5, "label": "Large network (128, 5)"}
]

plt.figure(figsize=(12, 8))
plt.scatter(X_numpy, y_numpy, color='blue', alpha=0.2, s=10, label='Noisy data')
plt.plot(X_numpy, np.sin(X_numpy), color='green', linewidth=3, label='True sin function')

colors = ['red', 'orange', 'purple']
for i, config in enumerate(configs):
    # 创建并训练模型
    model = SinNet(hidden_size=config["hidden_size"], num_layers=config["num_layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 快速训练
    for epoch in range(500):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 预测并绘制
    with torch.no_grad():
        y_predicted = model(X).numpy()

    plt.plot(X_numpy, y_predicted, color=colors[i], linewidth=2, label=config["label"])

plt.title('Comparison of different network architectures')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
