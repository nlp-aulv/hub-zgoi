
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

hidden_size = 64
# 2. 直接创建参数张量 a 和 b
# 这里是主要修改的部分，我们不再使用 nn.Linear。
# torch.randn() 生成随机值作为初始值。
# requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
W1 = torch.randn(1, hidden_size, requires_grad=True, dtype=torch.float)
b1 = torch.randn(hidden_size, requires_grad=True, dtype=torch.float)

W2 = torch.randn(hidden_size, 1, requires_grad=True, dtype=torch.float)
b2 = torch.randn(1, requires_grad=True, dtype=torch.float)

params = [W1, b1, W2, b2]

print("参数初始化完成")
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss()

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
# 这边试下来Adam的效果比SGD好一些
optimizer = torch.optim.Adam(params, lr=0.01)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    hidden = torch.relu(X @ W1 + b1)
    y_pred = hidden @ W2 + b2  # 输出层（线性）

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
with torch.no_grad():
    h = torch.relu(X @ W1 + b1)
    y_predicted = h @ W2 + b2

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.scatter(X_numpy, y_predicted, label=f'Model', color='red', alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('./res.png')
