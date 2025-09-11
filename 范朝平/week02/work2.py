import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置matplotlib参数，解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成正弦函数数据
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # 添加一些噪声

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("正弦函数数据生成完成。")
print(f"数据形状: X: {X.shape}, y: {y.shape}")
print("---" * 10)


# 2. 定义神经网络模型
class SinNet(nn.Module):
    def __init__(self, hidden_size=64):
        super(SinNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)


# 3. 初始化模型、损失函数和优化器
model = SinNet(hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("模型结构:")
print(model)
print("---" * 10)

# 4. 训练模型
num_epochs = 5000
losses = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每500个epoch打印一次损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 绘制训练损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')  # 使用对数刻度更好地显示损失下降

# 6. 绘制拟合结果
plt.subplot(1, 2, 2)
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.scatter(X_numpy, y_numpy, label='Noisy sin(x) data', color='blue', alpha=0.3, s=10)
plt.plot(X_numpy, y_predicted, label='Neural network fit', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting sin(x) with Neural Network')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n训练完成！")
print("---" * 10)

# 7. 在测试集上评估
X_test = np.linspace(-3 * np.pi, 3 * np.pi, 1000).reshape(-1, 1)
X_test_tensor = torch.from_numpy(X_test).float()

with torch.no_grad():
    y_test_pred = model(X_test_tensor).numpy()

plt.figure(figsize=(10, 6))
plt.plot(X_test, np.sin(X_test), label='True sin(x)', color='green', linestyle='--', linewidth=2)
plt.plot(X_test, y_test_pred, label='Neural network prediction', color='red', linewidth=2)
plt.scatter(X_numpy, y_numpy, label='Training data', color='blue', alpha=0.3, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generalization on Extended Domain')
plt.legend()
plt.grid(True)
plt.show()

# 8. 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total_params}")