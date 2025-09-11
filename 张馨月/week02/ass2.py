"""
调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
"""

import math
import random
import os
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# 0) 基础设置（可复现 + 设备）
# -----------------------
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

# -----------------------
# 1) 数据：y=sin(x), x∈[0, 2π]
# -----------------------
def make_sin_data(n_samples: int = 1024, train_ratio: float = 0.8):
    xs = np.linspace(0.0, 2.0 * math.pi, num=n_samples, dtype=np.float32)
    ys = np.sin(xs, dtype=np.float32)

    # 划分训练/验证
    n_train = int(n_samples * train_ratio)
    idxs = np.arange(n_samples)
    np.random.shuffle(idxs)
    train_idx, val_idx = idxs[:n_train], idxs[n_train:]

    x_train, y_train = xs[train_idx], ys[train_idx]
    x_val,   y_val   = xs[val_idx],   ys[val_idx]

    # 转 tensor
    x_train = torch.from_numpy(x_train).view(-1, 1).to(device)
    y_train = torch.from_numpy(y_train).view(-1, 1).to(device)
    x_val   = torch.from_numpy(x_val).view(-1, 1).to(device)
    y_val   = torch.from_numpy(y_val).view(-1, 1).to(device)
    return x_train, y_train, x_val, y_val

# -----------------------
# 2) 可配置的 MLP
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_layers: List[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------
# 3) 训练循环
# -----------------------
def train_model(model: nn.Module,
                x_train: torch.Tensor, y_train: torch.Tensor,
                x_val: torch.Tensor, y_val: torch.Tensor,
                epochs: int = 200,
                lr: float = 1e-3,
                batch_size: int = 64) -> Dict[str, List[float]]:
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = x_train.size(0)
    history = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        train_loss_acc = 0.0
        nb = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = x_train[idx]
            yb = y_train[idx]

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_acc += loss.item()
            nb += 1

        mean_train_loss = train_loss_acc / max(nb, 1)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = loss_fn(val_pred, y_val).item()

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_loss)

    return history

# -----------------------
# 4) 运行：两层隐藏层 [64, 32]
# -----------------------
def main():
    x_train, y_train, x_val, y_val = make_sin_data(n_samples=1024, train_ratio=0.8)
    model = MLP(in_dim=1, hidden_layers=[64, 32], out_dim=1)
    hist = train_model(model, x_train, y_train, x_val, y_val,
                       epochs=200, lr=1e-3, batch_size=64)

    # 最终 Loss
    final_train_loss = hist["train_loss"][-1]
    final_val_loss = hist["val_loss"][-1]
    print("【sin拟合：两层MLP [64, 32]】")
    print(f"最终训练Loss(MSE): {final_train_loss:.6f}")
    print(f"最终验证Loss(MSE): {final_val_loss:.6f}")

    # (1) 训练/验证Loss曲线（单图）
    plt.figure()
    plt.plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"], label="Train Loss")
    plt.plot(range(1, len(hist["val_loss"]) + 1), hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("sin(x) 拟合：训练/验证Loss曲线（MLP[64,32]）")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/sin_fit_loss_curve.png", dpi=150)
    plt.show()

    # (2) 真实 vs 预测（单图）
    with torch.no_grad():
        xs_plot = torch.linspace(0.0, 2.0 * math.pi, steps=600, device=device).view(-1, 1)
        ys_true = torch.sin(xs_plot)
        ys_pred = model(xs_plot)

    xs_np = xs_plot.cpu().numpy().ravel()
    ys_true_np = ys_true.cpu().numpy().ravel()
    ys_pred_np = ys_pred.cpu().numpy().ravel()

    plt.figure()
    plt.plot(xs_np, ys_true_np, label="True sin(x)")
    plt.plot(xs_np, ys_pred_np, label="Model Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin(x) 真实曲线 vs 模型预测（MLP[64,32]）")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/sin_fit_prediction_vs_true.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
