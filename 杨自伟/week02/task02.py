import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 生成模拟数据 (与之前相同)
# X_numpy = np.random.rand(100, 1) * 10
# y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)
x_numpy = np.random.rand(100, 1) * 10 *np.pi
y_numpy = 2 * np.sin(x_numpy) + 1 # sin函数
X = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 直接创建参数张量 a 和 b
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
# help(torch.randn)

print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
print("---" * 10)

# 3. 定义学习率
learning_rate = 0.001 # 学习率

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：计算预测值 y_pred
    y_pred = a * X + b

    # 手动计算 MSE 损失
    loss = torch.mean((y_pred - y)**2)

    # 手动反向传播：计算 a 和 b 的梯度
    # PyTorch 的自动求导会帮我们计算，我们只需要调用 loss.backward()
    # 但在这里，我们手动计算梯度，因此需要确保梯度清零
    if a.grad is not None:
        a.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()

    loss.backward()

    # 手动更新参数
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
a_learned = a.item()
b_learned = b.item()
print(f"拟合的斜率 a: {a_learned:.4f}")
print(f"拟合的截距 b: {b_learned:.4f}")
print("---" * 10)

# 6. 绘制结果
with torch.no_grad():
    y_predicted = a_learned *np.sin(X) + b_learned # 预测函数

plt.figure(figsize=(10, 6))
plt.scatter(x_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(x_numpy, y_predicted, label=f'Model: y = {a_learned:.2f}sin(x) + {b_learned:.2f}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
