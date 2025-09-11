import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#拟合sinx函数

#生成数据
np.random.seed(42)
x = np.sort(np.random.uniform(-2*np.pi, 2*np.pi, 1000))
y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

#将数据转换为张量
x_tensor = torch.from_numpy(x).float().view(-1,1) #(batch_size,fetures)
y_tensor = torch.from_numpy(y).float().view(-1,1)

#定义神经网络模型
class SineNet(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(SineNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size,hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1,hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2,output_size)
        )

    def forward(self,x):
        return self.network(x)
    
#定义超参数
input_size = 1
hidden_size1 = 128
hidden_size2 = 128
output_size = 1

#实例化模型
model = SineNet(input_size,hidden_size1,hidden_size2,output_size)
#定义损失函数和优化器
criterion = nn.MSELoss()#均方误差损失
optimizer = optim.Adam(model.parameters(),lr=0.01)

#训练模型
epochs = 2000
for epoch in range(epochs):
    #训练模式
    model.train()
    
    #前向传播
    y_pred = model(x_tensor)
    #计算损失
    loss = criterion(y_pred,y_tensor)
    #反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f'epoch[{epoch+1}/{epochs}] loss:{loss.item()}')

model.eval()
with torch.no_grad():
    x_sorted,indices = torch.sort(x_tensor,dim=0)
    y_pred = model(x_sorted)


plt.figure(figsize=(12,6))
#真实sin函数
plt.plot(x, np.sin(x), label=f'true sin(x)', color='blue', linewidth=2)
#预测
plt.plot(x, y_pred.numpy(), label=f'predicted sin(x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
