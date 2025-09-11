import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据
X_numpy = (np.random.rand(300, 1) * 4 - 2) * np.pi  # 形状为 (300, 1) 的二维数组，其中包含 300 个在 (-2, 2) 范围内均匀分布的随机浮点数。
Y_numpy = np.sin(X_numpy) + np.random.rand(300, 1) * 0.5
X = torch.from_numpy(X_numpy).float()  # 转化为tensor
Y = torch.from_numpy(Y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 创建网络
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = SimpleNet()
print("创建网络完成。")
print("---" * 10)

# 3. 定义损失函数和优化器
criterion = torch.nn.MSELoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print("定义损失函数和优化器完成。")
print("---" * 10)

# 4. 训练模型
print("正在训练模型")
num_epochs = 1000 # 训练次数为1000次
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    output = model(X)
    loss = criterion(output, Y)

    loss.backward()
    optimizer.step()

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f"epoch: [{epoch + 1}/{num_epochs}], loss: {loss.item():.4f}")

print("训练完成")
print("---" * 10)

# 5. 使用模型进行预测
print("正在进行预测")
X_plot = np.linspace(-2*np.pi, 2*np.pi, 500)
X_plot_tensor = torch.from_numpy(X_plot).float().unsqueeze(1)  # 转化为tensor
model.eval()
with torch.no_grad():
    Y_plot_pred_tensor = model(X_plot_tensor)
Y_plot_pred = Y_plot_pred_tensor.numpy().squeeze()

print("预测完成")
print("---" * 10)

# 6. 绘制结果
plt.figure(figsize=(10, 6))
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X_numpy, Y_numpy, label='Raw data', color='blue', alpha=0.5)
plt.plot(X_plot, Y_plot_pred, label='Prediction', color='red', linewidth=2)
plt.legend()
plt.grid(True)
plt.show()
