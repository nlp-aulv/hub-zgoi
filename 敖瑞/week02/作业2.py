import matplotlib
matplotlib.use('TkAgg')
import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# 构造sin函数
def get_sin(num_samples=500, noise=0.03):
    x = np.linspace(-2.5 * np.pi, 2.5 * np.pi, num_samples)
    y = np.sin(x) + noise * np.random.randn(num_samples)
    return x, y


# 网络
class SinRegressor(nn.Module):
    def __init__(self, layer_size):
        super().__init__()

        layers = []
        for i in range(len(layer_size)-1):
            layers.append(nn.Linear(layer_size[i], layer_size[i+1]))
            if i < len(layer_size) - 2:
                layers.append(nn.ReLU())
        print('模型结构：', layers)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 训练
def train(layer_size, x, y):
    model = SinRegressor(layer_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_loss = []
    num_epoch = 2000
    for epoch in range(num_epoch):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        # print('Epoch: %d  Avg Loss: %f' % (epoch+1, np.mean(train_loss)))

    return model, train_loss


if __name__ == '__main__':
    x, y = get_sin(1000)
    X = torch.FloatTensor(x).reshape(-1, 1)
    y = torch.FloatTensor(y).reshape(-1, 1)

    layer_size = [1, 128, 32, 1]
    model, train_loss = train(layer_size, X, y)

    test_x = torch.linspace(X.min(), X.max(), 1000).reshape((-1, 1))
    true_y = np.sin(test_x)
    model.eval()
    with torch.no_grad():
        pred_y = model(test_x)
    # print(train_loss)

    errors = pred_y.numpy() - true_y.numpy()

    # 训练损失函数
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 预测数据
    plt.subplot(2, 2, 2)
    plt.plot(test_x.numpy(), pred_y.numpy(), label='Pred Data')
    plt.plot(test_x.numpy(), true_y.numpy(), label='True Data')
    plt.title('Pred & True Data')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 误差
    plt.subplot(2, 2, 4)
    plt.hist(errors, bins=30, alpha=0.7)
    plt.title('Errors')
    plt.grid(True)
    plt.show()
