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
lr = 1 # initial learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

### LR scheduling
lam = 0.1
print(f"gamma: {np.exp(-lam):.4f}")
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=np.exp(-lam)
)

loss_function = nn.MSELoss()
#%%
epochs = 100

xs, ys, loss_history, lr_history = [], [], [], []
for epoch in range(epochs):
    for x_batch, y_batch in loader:
        ### 현재 parameter의 value
        xs.append(model.linear.weight.detach().item())
        ys.append(model.linear.bias.detach().item())
        lr_history.append(optimizer.param_groups[0]["lr"])  # 현재 lr 기록
        
        optimizer.zero_grad()

        ### SGD
        y_hat = model(x_batch)
        loss = loss_function(y_batch, y_hat)
        loss_history.append(loss.detach().item())

        loss.backward()
        optimizer.step()

    ### exponential decay 적용 (epoch 기준)
    scheduler.step()
#%%
### 학습 경과에 따른 손실함수 확인
plt.figure(figsize=(6, 3.5))

plt.plot(loss_history, linewidth=2)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title("Training Loss Curve (SGD) with Exponential LR Decay", fontsize=13)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/3_loss_with_lr_scheduling.png")
plt.show()
plt.close()
#%%
plt.figure(figsize=(6, 3.5))

plt.plot(lr_history, linewidth=2, color="orange")
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Learning Rate", fontsize=12)
plt.title("LR with Exponential Decay",fontsize=13)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/3_lr_with_decay.png")
plt.show()
plt.close()
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
plt.contour(w1, b1, Z, levels=10, zorder=1)
plt.scatter(w_true, b_true, s=200, color='green', marker='*', zorder=3, edgecolors='black', linewidths=1.5)
plt.plot(xs, ys, linewidth=2, zorder=2)
plt.scatter(xs, ys, s=10, zorder=2)
plt.title(f"SGD (minibatch) with step size {lr}", fontsize=16)
plt.xlabel("w", fontsize=14)
plt.ylabel("b", fontsize=14)
plt.tight_layout()
plt.savefig("./fig/3_contour_with_lr_scheduling.png")
plt.show()
plt.close()
#%%