#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
#%%
### Reproducibility
seed = 42
torch.manual_seed(seed)
#%%
### 학습용 데이터셋 생성
# x ~ Uniform[-2pi, 2pi]
# y = sin(2x) + 0.5x
n = 5000
x = torch.rand(size=(n, 1))
x = x * 4 * torch.pi - 2 * torch.pi # Uniform[-2pi, 2pi]
y = torch.sin(2 * x) + 0.5 * x
y = y + 0.3 * torch.randn_like(y) # small noise
#%%
plt.figure(figsize=(7, 4))
plt.scatter(x, y, s=10, alpha=0.3, label="학습 데이터")
plt.xlabel("x", fontsize=15)
plt.ylabel("y", fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("./fig/3_traindata.png")
plt.show()
plt.close()
#%%
dataset = TensorDataset(x, y)
dataset.tensors
dataset[0] # 첫번째 관측치 쌍

loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)
#%%
### 신경망모형: 3개의 hidden layer를 갖는 MLP
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP() # default configuration 사용
model.train() # 학습모드
#%%
### 학습을 위한 준비
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#%%
### 모형 학습
epochs = 500
loss_history = []
for e in range(epochs):
    epoch_loss = []
    for x_batch, y_batch in loader:
        pred = model(x_batch) # 모형 계산 (forward)
        loss = loss_function(pred, y_batch) # loss function 계산

        optimizer.zero_grad() # gradient 초기화
        loss.backward() # gradient 계산
        optimizer.step() # SGD update

        epoch_loss.append(loss.item()) # 한 epoch내에서의 학습 기록
    loss_history.append(np.mean(epoch_loss)) # 매 epoch별 평균 손실함수 값 기록

    if e % 20 == 0:
        print(f"Epoch {e+1:3d} | MSE = {loss_history[-1]:.4f}") # 가장 마지막의 MSE를 출력
#%%
### 학습 진단 1
plt.figure(figsize=(6, 3))
plt.plot(loss_history, linewidth=2)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title("Training Loss Curve", fontsize=13)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/3_training_loss.png")
plt.show()
plt.close()
#%%
### 학습 진단 2
model.eval() # 평가모드
with torch.no_grad():
    x_pred = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100).view(-1, 1) # [100, 1]
    y_pred = model(x_pred) # [100, 1]

plt.figure(figsize=(7, 4))
# 학습 데이터
plt.scatter(
    x, y,
    s=10, alpha=0.3, label="학습 데이터",
)
# 모델 예측
plt.plot(
    x_pred, y_pred,
    linewidth=3, label="MLP 예측값", color='red'
)
plt.xlabel("x", fontsize=15)
plt.ylabel("y", fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("./fig/3_prediction.png")
plt.show()
plt.close()
#%%