import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据
X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)  # 生成数据范围为 0 到 2π
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)  # sin 函数数据 + 噪声
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层感知机模型
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 输入层到隐藏层 (1 输入特征, 64 隐藏单元)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleMLP()  # 创建模型实例

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    y_pred = model(X)  # 前向传播
    loss = loss_fn(y_pred, y)  # 计算损失
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 进行预测和可视化
model.eval()  # 设置为评估模式
with torch.no_grad():
    y_predicted = model(X).numpy()  # 获取模型预测

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw sin data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label='Fitted curve', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sin Function Fitting with MLP')
plt.legend()
plt.grid(True)
plt.show()
