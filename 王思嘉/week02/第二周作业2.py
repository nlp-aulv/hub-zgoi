import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


# 1. 生成模拟数据 (呈现sinx形状)
X_numpy = np.random.rand(100, 1) * 10              # 生成 100 个随机数，范围在 0 到 10 之间
y_numpy = np.sin(X_numpy) + 0.3 * np.random.randn(100, 1)  
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 这里定义多层神经网络模型
class SinNet(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[64, 32, 16], output_size=1):
        super(SinNet, self).__init__()
        # 创建网络层列表
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        # 将所有层组合成一个序列模型
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 实例化模型
model = SinNet(input_size=1, hidden_sizes=[256, 128, 64], output_size=1)#这里使用三个隐藏层训练
print("多层神经网络模型创建完成。")
print("---" * 10)    

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

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
    y_predicted = model(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.scatter(X_numpy, y_predicted, label='Fitted line', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
