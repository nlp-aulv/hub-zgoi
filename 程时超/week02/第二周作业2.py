import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成sin函数数据
np.random.seed(44)
X_numpy = np.random.rand(200, 1) * 4 * np.pi  # [0,4π)
y_numpy = np.sin(X_numpy) + 0.2 * np.random.randn(200, 1)  # 添加一些噪声

# 转换为torch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


# 定义全链接网络
class SinMLP(nn.Module):
    def __init__(self, hidden_size):
        super(SinMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)


model = SinMLP(hidden_size=64)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("模型结构:")
print(model)
print("---" * 10)

# 训练模型
num_epochs = 2000
losses = []

for epoch in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
print("---" * 10)

# 绘制结果
with torch.no_grad():
    X_test = torch.linspace(0, 4 * np.pi, 1000).reshape(-1, 1).float()
    y_pred_test = model(X_test)

plt.figure(figsize=(15, 5))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_test.numpy(), y_pred_test.numpy(), label='sin model', color='red', linewidth=2)
plt.plot(X_test.numpy(), np.sin(X_test.numpy()), label='true sin(x)', color='blue', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('y=sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# 打印最终损失
print(f"最终训练损失: {losses[-1]:.4f}")
