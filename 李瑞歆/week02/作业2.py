import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成 sin 数据 -------------------------------------------------------------
X_numpy = np.linspace(-np.pi, np.pi, 200, dtype=np.float32).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)

# X_numpy = np.random.rand(100, 1) * 10
# y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

# 2. 定义带非线性的极小网络 ----------------------------------------------------
# 输入 1 维 -> 隐藏 10 维 (tanh) -> 输出 1 维
model = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.Tanh(),          # 不选 ReLU 是因为 Tanh 更接近 sin 的曲线
    torch.nn.Linear(10, 1)
)

# 3. 损失和优化器 --------------------------------------------------------------
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练 -----------------------------------------------------------------------
num_epochs = 2000
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch {epoch+1:4d}/{num_epochs}, Loss = {loss.item():.6f}')

# 5. 绘图 -----------------------------------------------------------------------
with torch.no_grad():
    y_pred_np = y_pred.numpy()

plt.figure(figsize=(7, 4))
plt.scatter(X_numpy, y_numpy, s=8, label='Raw data')
plt.plot(X_numpy, y_pred_np, 'r', linewidth=2, label='Learned curve')
plt.legend(); plt.grid(); plt.show()