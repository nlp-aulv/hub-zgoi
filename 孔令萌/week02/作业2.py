import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 1. 生成sin函数的模拟数据
X_numpy = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成")
print(f"输入数据形状: {X.shape}")
print(f"输出数据形状: {y.shape}")
print("---" * 10)


# 2. 定义多层神经网络模型
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        # 定义多层全连接网络
        self.layers = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),  # 激活函数
            nn.Linear(32, 64),
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 32),
            nn.ReLU(),  # 激活函数
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 前向传播
        return self.layers(x)


# 初始化模型
model = SinNet()
print("多层神经网络结构:")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失，适用于回归任务
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 5000  # 训练轮次
losses = []  # 记录损失变化

for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # 反向传播和参数更新
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每500个epoch打印一次
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成!")
print("---" * 10)

# 5. 模型预测
model.eval()
with torch.no_grad():
    y_pred = model(X)

# 转换为numpy数组用于可视化
y_pred_numpy = y_pred.numpy()

# 6. 可视化结果
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_numpy, y_numpy, label='Noisy Data', color='blue', alpha=0.3, s=10)
plt.plot(X_numpy, np.sin(X_numpy), label='True Sin Function', color='green', linewidth=2)
plt.plot(X_numpy, y_pred_numpy, label='Model Prediction', color='red', linewidth=2)
plt.title('Sin Function Fitting Result')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
