import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 生成正弦函数模拟数据
# 生成[0,4π]范围内的数据
X_numpy = np.random.rand(100, 1) * 4 * np.pi
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)

X = torch.from_numpy(X_numpy).float()  # torch 中所有的计算通过tensor计算
y = torch.from_numpy(y_numpy).float()


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):  # 层的个数和验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        # 创建多层网络
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 模型参数设置
hidden_dims = [256, 128, 56]
print("隐藏层数", len(hidden_dims))
print("各层节点数", hidden_dims)

input_dim = 1  # 输入维度（x值）
output_dim = 1  # 输出维度（sin(x)值）

# 初始化模型
model = SimpleClassifier(input_dim, hidden_dims, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 回归问题使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 训练参数
num_epochs = 2000
batch_size = 32

# 创建数据加载器（如果需要批量训练）
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
losses = []
print("开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    losses.append(epoch_loss)

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}')

print("训练完成!")

# 测试模型
model.eval()
with torch.no_grad():
    # 生成测试数据
    test_X = torch.linspace(0, 4 * np.pi, 100).unsqueeze(1)
    predictions = model(test_X)
    true_y = torch.sin(test_X)

# 可视化结果
plt.figure(figsize=(15, 5))

# 1. 绘制损失曲线
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

# 2. 绘制训练数据和预测结果
plt.subplot(1, 3, 2)
plt.scatter(X_numpy, y_numpy, alpha=0.6, label='Training Data', s=20)
plt.plot(test_X.numpy(), predictions.numpy(), 'r-', linewidth=2, label='Model Prediction')
plt.plot(test_X.numpy(), true_y.numpy(), 'g--', linewidth=2, label='True sin(x)')
plt.title('Sin Function Approximation')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)

# 3. 绘制误差曲线
plt.subplot(1, 3, 3)
error = np.abs(predictions.numpy() - true_y.numpy())
plt.plot(test_X.numpy(), error, 'b-', linewidth=2)
plt.title('Prediction Error')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.grid(True)

plt.tight_layout()
plt.show()
