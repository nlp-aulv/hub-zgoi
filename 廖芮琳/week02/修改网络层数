import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


# 1. 生成模拟数据 (与之前相同)
# X_numpy = np.random.rand(100, 1) * 10
# y_numpy = np.sin(X_numpy) + np.random.randn(100, 1)
# X = torch.from_numpy(X_numpy).float()
# y = torch.from_numpy(y_numpy).float()

x = np.linspace(-2*np.pi, 2*np.pi, 800).reshape(-1, 1)
y = np.sin(x) + np.random.randn(800, 1) * 0.1

x_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y)

print("数据生成完成。")
print("---" * 10)

# 2. 直接创建参数张量 a 和 b
# 这里是主要修改的部分，我们不再使用 nn.Linear。
# torch.randn() 生成随机值作为初始值。
# requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
# a = torch.randn(1, requires_grad=True, dtype=torch.float)
# b = torch.randn(1, requires_grad=True, dtype=torch.float)
#
# print(f"初始参数 a: {a.item():.4f}")
# print(f"初始参数 b: {b.item():.4f}")
# print("---" * 10)

class SinNet(nn.Module):
    def __init__(self, hidden_dim=64, num_layer=3):
        super(SinNet, self).__init__()
        layers = []

        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layer - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


model = SinNet(hidden_dim=64, num_layer=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
# loss_fn = torch.nn.MSELoss()  # 回归任务

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
# optimizer = torch.optim.SGD([a, b], lr=0.01)

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    # y_pred = a * X + b

    model.train()

    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # 计算损失
    # loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
# a_learned = a.item()
# b_learned = b.item()
# print(f"拟合的斜率 a: {a_learned:.4f}")
# print(f"拟合的截距 b: {b_learned:.4f}")
print("---" * 10)


with torch.no_grad():
    x_full = torch.FloatTensor(x)
    y_predicted = model(x_full).numpy()

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
# with torch.no_grad():

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Raw data', color='blue', alpha=0.6)
plt.plot(x_tensor, y_predicted, label='Neural network approximation', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
