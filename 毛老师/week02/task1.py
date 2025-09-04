import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1) # 批量训练， 100*1
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) / 20

# 将NumPy数组转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()


# 定义线性回归模型
# nn.Linear(in_features, out_features)
# 在这里，输入和输出的特征数都为1
# model = nn.Linear(1, 1) # 全连接层

# 多层的前馈网络，全连接网络
# 层数，每层的神经元数量
# 简单 -》 复杂
# 前面的层提取特征 增加维度， 后面的层将特征转换为输出，可能是降低维度
# 1 -> 100 -> 100 -> 10 -> 1
model = nn.Sequential(
    nn.Linear(1, 100), # 随机初始化
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
)



# 定义损失函数 (均方误差)
loss_fn = nn.MSELoss()

# 定义优化器 (随机梯度下降)
# model.parameters() 会自动找到模型中需要优化的参数（即a和b）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr 是学习率

# 训练模型
num_epochs = 5000  # 训练迭代次数
for epoch in range(num_epochs): # 推荐每次epoch之后，原始X 和 label 次序打乱
    # 前向传播：计算预测值
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化：
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印最终学到的参数
# model.weight 是斜率a， model.bias 是截距b
print("\n训练完成！")
# a_learned = model.weight.item()
# b_learned = model.bias.item()
# print(f"拟合的斜率 a: {a_learned:.4f}")
# print(f"拟合的截距 b: {b_learned:.4f}")


# 将模型切换到评估模式，这在训练结束后是好习惯
model.eval() # 主动关闭dropout

# 禁用梯度计算，因为我们不再训练
with torch.no_grad():
    y_predicted = model(X).numpy() # 使用训练好的模型进行预测

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
