import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1,1)
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 64),  # 输入层 → 隐藏层 1 (64 神经元)
            torch.nn.ReLU(),  # 激活函数
            torch.nn.Linear(64, 32),  # 隐藏层 1 → 隐藏层 2 (32 神经元)
            torch.nn.ReLU(),  # 激活函数
            torch.nn.Linear(32, 1)  # 隐藏层 2 → 输出层 (1 神经元)
        )

    def forward(self, x):
        return self.net(x)


model = MLP()
print("模型结构：")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
# a * x + b 《 - 》  y'

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 2000
loss_history = []
for epoch in range(num_epochs):
    y_pred = model(X)  # 前向传播
    loss = loss_fn(y_pred, y)  # 计算损失
    loss_history.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_pred_numpy = model(X).numpy()

plt.figure(figsize=(12, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data (sin(x) + noise)', color='blue', alpha=0.5, s=10)
plt.plot(X_numpy, y_pred_numpy, label='MLP prediction', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('MLP Fitting sin(x)')
plt.grid(True)
plt.show()

# 绘制损失曲线
plt.figure(figsize=(12, 4))
plt.plot(loss_history, label='Training Loss', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()
