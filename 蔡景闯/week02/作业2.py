import torch
import numpy as np  # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
from torch import nn

# 设置 matplotlib 中文显示
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
# 1. 生成模拟数据
X_numpy = np.random.rand(100, 1) * 10

# Y = sin(X)
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.05  # 加入噪声
X = torch.from_numpy(X_numpy).float()  # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


# 2. 定义模型
class SinModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SinModel, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 实例化模型
model = SinModel()

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss()  # 回归任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器

# 4. 训练模型
num_epochs = 1000
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    loss_list = []
    for i in range(0, X.shape[0], batch_size):
        # 选中指定 batch 的数据
        batch_x, batch_y = X[i:i + batch_size], y[i:i + batch_size]
        # 前向传播：
        y_pred = model(batch_x)

        # 计算损失
        loss = loss_fn(y_pred, batch_y)
        loss_list.append(loss.item())
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(loss_list):.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model(X)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.scatter(X.numpy(), y.numpy(), label='真实值: sin(x)')
plt.scatter(X.numpy(), y_predicted.detach().numpy(), label='预测值: 模型输出')
plt.title('sin函数拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('sin函数拟合.png')
plt.show()
