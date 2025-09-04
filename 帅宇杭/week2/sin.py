import torch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

X_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + np.random.normal(0, 0.01, X_numpy.shape)

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X_numpy)

X = torch.from_numpy(X_poly).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成 done")

n_features = X.shape[1]
weights = torch.randn(n_features, 1, requires_grad=True, dtype=torch.float)

print(f"初始参数形状：{weights.shape}")

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD([weights], lr=0.05)

num_epochs = 5000
losses = []

for epoch in range(num_epochs):
    y_pred = torch.mm(X, weights)

    loss = loss_fn(y_pred, y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"epoch: [{epoch+1} / {num_epochs}], loss: {loss.item():.6f}")

print("训练完成")

with torch.no_grad():
    y_predicted = torch.mm(X, weights).numpy()

plt.figure(figsize=(12, 6))
plt.scatter(X_numpy, y_numpy, label="sin(x)", color="green", alpha=0.5)

x_true = np.linspace(0, 2 * np.pi, 1000)
y_true = np.sin(x_true)
plt.plot(x_true, y_true, label = "sin(x)", color="red", linewidth=2)

plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('y=sin(x)')
plt.legend()
plt.grid(True)
plt.show()

with torch.no_grad():
    coefficients = weights.numpy().flatten()

equation = "sin(x) ≈ "
for i, coef in enumerate(coefficients):
    power = i
    if power == 0:
        equation += f"{coef:.4f}"
    else:
        if coef >= 0:
            equation += f" + {coef:.4f}x^{power}"
        else:
            equation += f" - {abs(coef):.4f}x^{power}"

print("拟合的多项式方程:")
print(equation)
