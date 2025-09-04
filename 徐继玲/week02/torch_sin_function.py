import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 1. 生成正弦函数数据
x_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000)  # 生成-2π到2π之间的1000个点
y_numpy = np.sin(x_numpy)  # 计算对应的正弦值

# 转换为PyTorch张量
X = torch.from_numpy(x_numpy).float().unsqueeze(1)  # 增加一个维度 (1000, 1)
y = torch.from_numpy(y_numpy).float().unsqueeze(1)  # 增加一个维度 (1000, 1)

print(f"数据生成完成。共生成 {len(x_numpy)} 个数据点。")
print("---" * 10)


# 2. 定义多层感知机模型
class SinNet(nn.Module):
    def __init__(self, hidden_layers=[64, 64]):
        """
        定义用于拟合正弦函数的神经网络
        - 输入层: 1个神经元 (x值)
        - 隐藏层: 可配置的多层结构
        - 输出层: 1个神经元 (预测的sin(x)值)
        """
        super(SinNet, self).__init__()
        layers = []

        # 构建隐藏层
        input_size = 1
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # 使用ReLU激活函数
            input_size = hidden_size

        # 输出层
        layers.append(nn.Linear(input_size, 1))

        # 组合所有层
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 3. 创建模型、损失函数和优化器
model = SinNet(hidden_layers=[128, 64])  # 128和64个神经元的隐藏层
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

print("模型结构:")
print(model)
print("---" * 10)

# 4. 训练模型
num_epochs = 5000
loss_history = []  # 记录损失变化

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)
    loss_history.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每500个epoch打印一次损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成!")
print("---" * 10)

# 5. 使用训练好的模型进行预测
with torch.no_grad():
    # 生成更密集的点用于绘制平滑曲线
    x_test = np.linspace(-2 * np.pi, 2 * np.pi, 2000)
    X_test = torch.from_numpy(x_test).float().unsqueeze(1)

    # 预测
    y_pred = model(X_test)

    # 转换为numpy数组
    x_test = x_test
    y_pred = y_pred.numpy().flatten()
    y_true = np.sin(x_test)

# 6. 可视化结果
plt.figure(figsize=(14, 8))

# 绘制真实正弦曲线
plt.plot(x_test, y_true, label='True Sine Wave', color='blue', linewidth=2.5)

# 绘制模型预测曲线
plt.plot(x_test, y_pred, label='Model Prediction', color='red', linestyle='--', linewidth=2)

# 绘制训练数据点
plt.scatter(x_numpy, y_numpy, label='Training Data', color='green', alpha=0.4, s=15)

plt.xlabel('x', fontsize=14)
plt.ylabel('sin(x)', fontsize=14)
plt.title('Neural Network Fitting of Sine Function', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.axis([-2 * np.pi, 2 * np.pi, -1.5, 1.5])

# 添加π刻度
pi_ticks = [-2 * np.pi, -1.5 * np.pi, -np.pi, -0.5 * np.pi, 0,
            0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi]
pi_labels = ['-2π', '-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π']
plt.xticks(pi_ticks, pi_labels, fontsize=12)
plt.yticks(fontsize=12)

# 绘制损失曲线
plt.figure(figsize=(12, 5))
plt.plot(loss_history, color='purple')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss History', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')  # 对数坐标更清晰展示损失变化
plt.show()