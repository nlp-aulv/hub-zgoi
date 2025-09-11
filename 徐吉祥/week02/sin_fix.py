import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#获取数据集
x_num = np.linspace(-np.pi*2, np.pi*2, 1000)
y_num = np.sin(x_num)
x = x_num.reshape(-1, 1)
y = y_num.reshape(-1, 1)
np.random.seed(42)
noise = np.random.normal(0, 0.05, y.shape)
y = y + noise
x_tensor = torch.FloatTensor(x).to(device)
y_tensor = torch.FloatTensor(y).to(device)
dataset = TensorDataset(x_tensor, y_tensor)

dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

#模型构建与训练
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32,1)
).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
for i in range(1000):
    # for x, y in dataloader:
    #     x, y = x.to(device), y.to(device)
        y_pre = model(x_tensor)
        loss = loss_fn(y_pre, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Loss: {loss.item():.4f}')


#模型结果测试与验证
x_test = np.linspace(-np.pi*2, np.pi*2, 1000)
y_test = np.sin(x_test)
model.eval()
model.to("cpu")
with torch.no_grad():
    y_pre = model(torch.FloatTensor(x_test.reshape(-1, 1))).detach().numpy().reshape(-1)

plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, label='y_target')
plt.plot(x_test, y_pre, label='y_input')
plt.show()