import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
X_numpy = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)  # 0-2π区间
y_numpy = np.sin(X_numpy)  # 目标函数

# 添加噪声
noise = np.random.normal(0, 0.1, y_numpy.shape)
y_numpy += noise

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()
print(f"生成{len(X)}个sin函数数据点，含高斯噪声")
print("---" * 10)


# 2. 定义多层神经网络
class SinModel(nn.Module):
    def __init__(self):
        super(SinModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),  # 输入层（1维）-> 隐藏层（64维）
            nn.Tanh(),  # 激活函数（用Tanh更适合周期函数）
            nn.Linear(64, 64),  # 隐藏层 -> 隐藏层
            nn.Tanh(),
            nn.Linear(64, 1)  # 隐藏层 -> 输出层（1维）
        )

    def forward(self, x):
        return self.net(x)


# 3. 初始化模型、损失函数和优化器
model = SinModel()
loss_fn = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

print("模型架构：")
print(model)
print("---" * 10)

# 4. 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每500个epoch打印损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 可视化结果
model.eval()  # 设置为评估模式
with torch.no_grad():
    y_pred = model(X).numpy()

plt.figure(figsize=(12, 6))
plt.scatter(X_numpy, y_numpy, s=10, label='Noisy sin(x)', alpha=0.6)
plt.plot(X_numpy, np.sin(X_numpy), 'g--', label='True sin(x)', linewidth=2)
plt.plot(X_numpy, y_pred, 'r-', label='Model prediction', linewidth=2)
plt.title('Multi-layer NN fitting sin function')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# 可视化隐藏层激活
sample_input = torch.linspace(0, 2 * np.pi, 100).float().unsqueeze(1)
with torch.no_grad():
    activations = []
    x = sample_input
    for layer in model.net:
        x = layer(x)
        if isinstance(layer, nn.Tanh):
            activations.append(x.numpy())

# 绘制隐藏层激活
plt.figure(figsize=(12, 8))
for i, act in enumerate(activations):
    plt.subplot(len(activations), 1, i + 1)
    plt.plot(sample_input.numpy(), act[:, 0:5])  # 只显示前5个神经元
    plt.title(f'Hidden Layer {i + 1} Activations (Tanh)')
plt.tight_layout()
plt.show()
