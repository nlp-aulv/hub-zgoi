import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.linspace(-2*np.pi, 2*np.pi, 1000)
# 加了一点噪音
zaoyin = 0.001
# 生成sin函数
y_numpy = np.sin(X_numpy)+zaoyin
X = torch.from_numpy(X_numpy).float().unsqueeze(1) # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float().unsqueeze(1)
print("数据生成完成。")
print("---" * 10)
class SinMore(nn.Module):
    # //这个函数有更优雅的写法，目前根据源代码修改成这个写法是最直观的
    def __init__(self, input_dim, hidden_dims, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SinMore, self).__init__()
        self.network = nn.Sequential(
            # 第1层：从 input_dim 到 hidden_dims[2]
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
            nn.ReLU(),  # 增加模型的复杂度，非线性
            # 第2层：从 hidden_dims 到 hidden_size2
            nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1]),
            nn.ReLU(),
            # 第3层：从 hidden_dims 到 hidden_size3
            nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[2]),
            nn.ReLU(),
            # 输出层：从 hidden_dims 到 output_size
            nn.Linear(in_features=hidden_dims[2], out_features=output_dim)
        )
        # 存储层信息
        self.hidden_dims = hidden_dims
    def forward(self, x):
        return self.network(x)
# 这个地方开始使用一个随便设置的值，结果一直报错后面改成1后就好了
input_dim =1
hidden_dims =[128,64,265,32]
# 输出是1
output_dim = 1
#调用模型
model_more = SinMore(input_dim, hidden_dims, output_dim) # 维度和精度有什么关系？
# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。torch.nn.MSELoss()
criterion_more = torch.nn.MSELoss()  # 回归任务
# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
#optimizer = torch.optim.SGD([a, b], lr=0.05) # 优化器，基于 a b 梯度 自动更新
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer_more = optim.SGD(model_more.parameters(), lr=0.01)

# 4. 创建DataLoader
dataset = TensorDataset(X, y)
print('打印下datasat:',dataset)
print(dataset[0])
print(dataset[1])
dataloader_more = DataLoader(dataset, batch_size=32, shuffle=True)
# 5. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model_more.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader_more):
        optimizer_more.zero_grad()
        outputs = model_more(inputs)
        loss = criterion_more(outputs, labels)
        loss.backward()
        optimizer_more.step()
        running_loss += loss.item()
        if idx % 100 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader_more):.4f}")
# 6. 打印，先把模型清空
model_more.eval()
print("\n训练完成！")

# 7. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
        y_predicted = model_more(X)
a_learned  = criterion_more(y_predicted, y).item()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='True sin(x)', color='blue', alpha=0.6)
# 这个地方使用中文后报错了
plt.plot(X_numpy, y_predicted, label=f'Model:More net sin(x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()



