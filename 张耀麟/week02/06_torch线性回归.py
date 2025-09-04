import torch
import numpy as np
import matplotlib.pyplot as plt
from SimpleClassifier import SimpleClassifier


def draw_pic(X, y_label, y_predict):
    plt.figure(figsize=(10, 6))
    plt.plot(X, y_label, label='Raw data', color='blue', alpha=0.6)
    plt.plot(X, y_predict, label=f'Predicted data', color='red', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 0. 超参数
    useSGD = False  # 修改这里切换SGD的超参数

    num_points, input_dims, output_dims, num_epochs = 0, [], [], 0
    if useSGD:
        num_points = 10000
        input_dims = [num_points, 10, 100]
        output_dims = [10, 100, num_points]
        num_epochs = 10000
    else:
        num_points = 10000
        input_dims = [num_points, 10]
        output_dims = [10, num_points]
        num_epochs = 400

    # 1. 生成数据
    X = torch.linspace(-torch.pi, torch.pi, num_points, dtype=torch.float)
    y_label = torch.sin(X)

    # 2. 构建模型
    model = SimpleClassifier(input_dims=input_dims, output_dims=output_dims)

    # 3. 定义损失函数和优化器
    loss_fn = torch.nn.MSELoss()
    if useSGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 4. 训练模型
    model.train()
    for epoch in range(num_epochs):
        # 前向传播
        y_predict = model(X)

        # 计算损失
        loss = loss_fn(y_predict, y_label)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        # 每100个 epoch 打印一次损失
        if (epoch + 1) % (num_epochs // 10) == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}')

    # 5. 绘制结果
    model.eval()
    with torch.no_grad():
        y_predicted = model(X)
        draw_pic(X, y_label, y_predicted)
