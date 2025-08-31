import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.animation as animation
from IPython.display import HTML

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)


# 生成训练数据 np.random.uniform 的作用是从一个均匀分布的区间 [low, high) 中随机抽取样本
def generate_data(n_samples=1000):
    """生成正弦函数训练数据"""
    x = np.random.uniform(-2 * np.pi, 2 * np.pi, n_samples)  # # 在[-2π, 2π]均匀采样
    y = np.sin(x)  # 计算对应的正弦值
    return x, y


# 定义多层感知机模型 模型架构： 输入(1) → 线性层(64) → ReLU → 线性层(64) → ReLU → 线性层(64) → ReLU → 输出(1)
class MLP(nn.Module):
    def __init__(self, hidden_size=64, num_layers=3):
        super(MLP, self).__init__()
        layers = []

        # 输入层: 1个输入特征 -> hidden_size个神经元
        layers.append(nn.Linear(1, hidden_size))
        layers.append(nn.ReLU())  # 激活函数

        # 隐藏层: hidden_size -> hidden_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # 输出层: hidden_size -> 1个输出
        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)  # 按顺序组合各层

    def forward(self, x):
        return self.network(x)


# 训练函数
def train_model(model, x_train, y_train, x_val, y_val, epochs=1000, learning_rate=0.001):
    """训练模型"""
    # 转换为PyTorch张量
    x_train_tensor = torch.FloatTensor(x_train).unsqueeze(1)  # 形状从 [n] -> [n, 1]  (1000,) -> ([1000, 1])
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    x_val_tensor = torch.FloatTensor(x_val).unsqueeze(1)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练过程中的损失
    train_losses = []
    val_losses = []

    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        model.train()

        # 前向传播
        predictions = model(x_train_tensor)  # 前向传播
        loss = criterion(predictions, y_train_tensor)  # 计算损失

        # 反向传播和优化
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 验证模式
        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor)

        # 记录损失
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # 每100个epoch打印一次损失
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

    return train_losses, val_losses


# 可视化函数
def visualize_results(model, x_train, y_train, train_losses, val_losses):
    """可视化训练结果"""
    # 生成测试数据（均匀分布，用于绘制平滑曲线）
    x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y_test = np.sin(x_test)

    # 转换为PyTorch张量
    x_test_tensor = torch.FloatTensor(x_test).unsqueeze(1)

    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(x_test_tensor).numpy().flatten()

    # 创建图形
    plt.figure(figsize=(15, 10))

    # 子图1: 真实值与预测值对比
    plt.subplot(2, 2, 1)
    plt.scatter(x_train, y_train, alpha=0.5, label='Training Data', s=10, color='gray')
    plt.plot(x_test, y_test, 'r-', label='True Sine Function', linewidth=2)
    plt.plot(x_test, predictions, 'b--', label='Model Predictions', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Sine Function Approximation')
    plt.legend()
    plt.grid(True)

    # 子图2: 损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # 使用对数刻度更好地显示损失变化

    # 子图3: 误差分布
    plt.subplot(2, 2, 3)
    errors = np.abs(predictions - y_test)
    plt.hist(errors, bins=50, alpha=0.7, color='green')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)

    # 子图4: 残差图
    plt.subplot(2, 2, 4)
    residuals = predictions - y_test
    plt.scatter(x_test, residuals, alpha=0.5, s=10, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 打印最终误差统计
    print(f"Final MSE: {np.mean(errors ** 2):.6f}")
    print(f"Final MAE: {np.mean(errors):.6f}")
    print(f"Max Absolute Error: {np.max(errors):.6f}")


# 创建训练过程动画
def create_training_animation(model, x_train, y_train, x_test, y_test, train_losses, val_losses):
    """创建训练过程动画"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 初始化模型用于动画
    anim_model = MLP(hidden_size=64, num_layers=3)
    optimizer = optim.Adam(anim_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    x_train_tensor = torch.FloatTensor(x_train).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    x_test_tensor = torch.FloatTensor(x_test).unsqueeze(1)

    # 初始化动画帧
    frames = []
    n_frames = 50
    step = len(train_losses) // n_frames

    for epoch in range(0, len(train_losses), step):
        # 训练到当前epoch
        if epoch > 0:
            for _ in range(step):
                anim_model.train()
                predictions = anim_model(x_train_tensor)
                loss = criterion(predictions, y_train_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 获取当前预测
        anim_model.eval()
        with torch.no_grad():
            current_predictions = anim_model(x_test_tensor).numpy().flatten()

        # 创建当前帧
        ax1.clear()
        ax2.clear()

        # 左图：函数拟合
        ax1.scatter(x_train, y_train, alpha=0.3, s=10, color='gray')
        ax1.plot(x_test, y_test, 'r-', label='True Sine', linewidth=2)
        ax1.plot(x_test, current_predictions, 'b--', label='MLP Prediction', linewidth=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('sin(x)')
        ax1.set_title(f'Epoch {epoch}')
        ax1.legend()
        ax1.grid(True)

        # 右图：损失曲线
        ax2.plot(train_losses[:epoch + 1], label='Training Loss')
        ax2.plot(val_losses[:epoch + 1], label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('Training Progress')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')

        frames.append([ax1, ax2])

    plt.tight_layout()
    return fig, frames


# 主函数
def main():
    # 生成数据
    print("Generating data...")
    x, y = generate_data(2000)

    # 划分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # 创建模型
    print("Creating model...")
    model = MLP(hidden_size=64, num_layers=3)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # 训练模型
    print("Training model...")
    train_losses, val_losses = train_model(
        model, x_train, y_train, x_val, y_val,
        epochs=2000, learning_rate=0.001
    )

    # 可视化结果
    print("Visualizing results...")
    visualize_results(model, x_train, y_train, train_losses, val_losses)

    # 生成测试数据用于动画
    x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y_test = np.sin(x_test)


if __name__ == "__main__":
    main()