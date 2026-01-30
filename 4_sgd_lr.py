#%%
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
#%%
torch.manual_seed(42)
#%%
### 가상의 훈련 데이터 생성
n = 200
x = torch.linspace(-1, 1, n).unsqueeze(1) # [n, 1] 차원
w_true, b_true = 1, 1
noise = 0.5 * torch.randn(n, 1) # noise ~ N(0, 0.5^2)
y = w_true * x + b_true + noise # y = wx + b + noise

batch_size = 64 # minibatch size
dataset = TensorDataset(x, y)
loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=False)
#%%
### 신경망모형 정의
class LinReg(nn.Module):
    def __init__(self, indim=1, outdim=1):
        super().__init__()
        self.linear = nn.Linear(in_features=indim, out_features=outdim)
        
    def forward(self, x):
        return self.linear(x)

model = LinReg()
model.train()
#%%
lr = 0.1 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

loss_function = nn.MSELoss()
#%%
epochs = 10

xs, ys, loss_history = [], [], []
for epoch in range(epochs):
    for x_batch, y_batch in loader:
        ### 현재 parameter의 value
        xs.append(model.linear.weight.detach().item())
        ys.append(model.linear.bias.detach().item())
        
        optimizer.zero_grad()

        ### SGD
        y_hat = model(x_batch)
        loss = loss_function(y_batch, y_hat)
        loss_history.append(loss.detach().item())

        loss.backward()
        optimizer.step()
#%%
### 학습 경과에 따른 손실함수 확인
plt.figure(figsize=(6, 3.5))

plt.plot(loss_history, linewidth=2)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title("Training Loss Curve (SGD)", fontsize=13)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
### contour plot
w1 = np.arange(-0.5, 2.5, 0.02) # 150개의 값
b1 = np.arange(-0.5, 2.5, 0.02) # 150개의 값
W, B = np.meshgrid(w1, b1)

Wt = torch.tensor(W, dtype=torch.float32).unsqueeze(-1) # [150, 150, 1]
Bt = torch.tensor(B, dtype=torch.float32).unsqueeze(-1) # [150, 150, 1]

x_ = x.squeeze(1).view(1, 1, -1) # [n, 1] --> [n, 1, 1] --> [1, 1, n]
y_ = y.squeeze(1).view(1, 1, -1) # [n, 1] --> [n, 1, 1] --> [1, 1, n]

y_hat_grid = Wt * x_ + Bt # w, b의 모든 grid 값에 대해서 y = wx + b를 계산
Z = torch.mean((y_hat_grid - y_) ** 2, dim=-1).numpy() # grid의 모든 y = wx + b에 대해서 MSE를 계산

plt.figure(figsize=(5, 5))
plt.contour(w1, b1, Z, levels=10)
plt.scatter(w_true, b_true, s=200, color='green', marker='*')
plt.plot(xs, ys, linewidth=2)
plt.scatter(xs, ys, s=10)
plt.title(f"SGD (minibatch) with step size {lr}")
plt.xlabel("w")
plt.ylabel("b")
plt.tight_layout()
plt.show()
#%%