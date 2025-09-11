import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim, tensor

np.random.seed(22)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.uniform(-3 * np.pi, 3 * np.pi, 2000)
y_numpy = np.sin(X_numpy)
y_numpy_label = y_numpy + np.random.uniform(-1, 1, 2000)
X = torch.from_numpy(X_numpy).float()
Y = torch.from_numpy(y_numpy_label).float()
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, color='b')
plt.scatter(X_numpy, y_numpy_label, color='y')
plt.xlim(-4 * np.pi, 4 * np.pi)
plt.ylim(-1.5, 1.5)
plt.show()
print("数据生成完成。")
print("---" * 10)


# 模型搭建
class Logic_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, is_dropout=False):
        super(Logic_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.is_dropout = is_dropout

    def forward(self, x):
        out = self.relu(self.fc1(x))
        if self.is_dropout:
            out = self.dropout(out)
        out = self.relu1(self.fc2(out))
        if self.is_dropout:
            out = self.dropout1(out)
        out = self.fc3(out)
        return out


# 构建模型训练
models = [Logic_Model(input_size=1, hidden_size=64, output_size=1),
          Logic_Model(input_size=1, hidden_size=64, output_size=1, is_dropout=True)]
optimizer = [optim.Adam(models[i].parameters(), lr=0.001) for i in range(len(models))]
loss_fn = nn.MSELoss()
# 训练模型
num_epochs = 1000
batch_size = 64
loss_history = []
for model in models:
    model.train()
    loss_history.append([])
for epoch in range(num_epochs):
    sum_loss = [0.0, 0.0]
    n_batches = len(X) // batch_size  # 计算批次数量
    for batch_idx in range(n_batches):
        indices = np.random.choice(len(X), size=batch_size, replace=False)
        x_batch = X[indices].unsqueeze(1)
        y_batch = Y[indices].unsqueeze(1)
        for i, model in enumerate(models):
            # 前向传播
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            # 反向传播和优化
            optimizer[i].zero_grad()
            loss.backward()
            optimizer[i].step()
            sum_loss[i] += loss.item()
    for i, loss in enumerate(sum_loss):
        loss_history[i].append(loss / n_batches)
    if epoch % 10 == 0:
        avg_loss = sum_loss[0] / n_batches
        avg_loss1 = sum_loss[1] / n_batches
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
        print(f'drop Epoch {epoch}, Average Loss: {avg_loss1:.4f}')

# 验证模型
model.eval()
with torch.no_grad():
    # 扩大验证范围
    X_numpy_verify = np.linspace(-3 * np.pi, 3 * np.pi, 2000)
    # 使用模型进行预测
    y_pred_numpy = models[0](torch.from_numpy(X_numpy_verify).float().unsqueeze(1)).numpy()
    y_pred_numpy_dis = models[1](torch.from_numpy(X_numpy_verify).float().unsqueeze(1)).numpy()
    # sin数据
    y_single_numpy = np.sin(X_numpy_verify)
    # 绘制预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X_numpy, y_numpy_label, color='g', label='Y_train')
    plt.scatter(X_numpy_verify, y_single_numpy, color='b', label='sin(x)')
    plt.scatter(X_numpy_verify, y_pred_numpy, color='r', label='model(x)')
    plt.scatter(X_numpy_verify, y_pred_numpy_dis, color='y', label='model(x) disDropout')
    plt.xlim(-4 * np.pi, 4 * np.pi)
    plt.ylim(-1.5, 1.5)
    plt.title('Model Prediction vs. sin(x)')
    plt.legend()
    plt.show()
# 绘制损失函数曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_history[0], label='Loss', color='r')
plt.plot(loss_history[1], label='Loss with Dropout', color='y')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
