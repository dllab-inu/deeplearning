#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
#%%
### 데이터 불러오기
df = pd.read_csv("./data/climate.csv", parse_dates=['timestep'], index_col=0)
print(df.shape)
print(df.head())
#%%
configs = {
    'seed': 42,

    'lookback': 28,
    'horizon': 7,
    'val_ratio': 0.2,

    'feature_dim': 1, # 단변량
    'hidden_dim': 16, # GRU hidden dimension
    'num_layers': 2, # GRU 레이어 수

    'epochs': 30,
    'batch_size': 128,
    'lr': 0.001,
}

### Reproducibility
torch.manual_seed(configs['seed'])
np.random.seed(configs['seed'])
#%%
### train, val / test split
trainval_len = len(df.loc[df['timestep'] < "2016-01-01"])
trainval = df.iloc[:trainval_len]
test = df.iloc[trainval_len - configs["lookback"]:] ### test dataset의 첫 번째 sequence의 input
trainval.tail(30)
test.head(30)

### train / val split
train_len = int(trainval_len * (1 - configs["val_ratio"]))
train = trainval.iloc[:train_len]
val = trainval.iloc[train_len - configs["lookback"]:] ### val dataset의 첫 번째 sequence의 input
print(train.shape)
print(val.shape)
print(test.shape)
#%%
### 단변량 시계열 데이터 - T (degC)
train_y = train[["T (degC)"]].values.astype("float32") # [T, 1]
train_y = torch.FloatTensor(train_y)

val_y = val[["T (degC)"]].values.astype("float32") # [T, 1]
val_y = torch.FloatTensor(val_y)

test_y = test[["T (degC)"]].values.astype("float32") # [T, 1]
test_y = torch.FloatTensor(test_y)
#%%
### 표준화 - 학습데이터의 통계량만을 이용
mean = train_y.mean()
std = train_y.std() + 1e-6
print(f"평균: {mean:.3f}, 표준편차: {std:.3f}")

train_y = (train_y - mean) / std
val_y = (val_y - mean) / std
test_y = (test_y - mean) / std
#%%
### window format dataset
    # dimension=0 - 시간 축 방향으로 슬라이딩
    # size - 윈도우 한 개의 길이
    # step = 1 - 한 timestep씩 이동
window_len = configs['lookback'] + configs['horizon']

train_y = train_y.unfold(
    dimension=0, size=window_len, step=1
).permute(0, 2, 1) # [T, 1, lookback+horizon] --> [T, lookback+horizon, 1]

val_y = val_y.unfold(
    dimension=0, size=window_len, step=1
).permute(0, 2, 1)

test_y = test_y.unfold(
    dimension=0, size=window_len, step=1
).permute(0, 2, 1)

print(train_y.shape) # [T, lookback+horizon, 1]
print(val_y.shape)
print(test_y.shape)
#%%
### torch dataset
train_dataset = TensorDataset(
    train_y[:, :configs['lookback'], :], # [T, lookback, 1]
    train_y[:, configs['lookback']:, :], # [T, horizon, 1]
)
val_dataset = TensorDataset(
    val_y[:, :configs['lookback'], :],
    val_y[:, configs['lookback']:, :],
)
test_dataset = TensorDataset(
    test_y[:, :configs['lookback'], :],
    test_y[:, configs['lookback']:, :],
)
#%%
### torch dataloader
train_loader = DataLoader(
    train_dataset, batch_size=configs['batch_size'], shuffle=True, drop_last=False)
val_loader = DataLoader(
    val_dataset, batch_size=configs['batch_size'], shuffle=False, drop_last=False)
test_loader = DataLoader(
    test_dataset, batch_size=configs['batch_size'], shuffle=False, drop_last=False)

x_batch, y_batch = next(iter(train_loader))

print("x_batch:", x_batch.shape)  # [B, lookback, 1]
print("y_batch:", y_batch.shape)  # [B, horizon, 1]
#%%
### seq2seq with GRU 
class Forecaster(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # encoder
        self.encoder = nn.GRU(
            input_size = configs['feature_dim'],
            hidden_size = configs['hidden_dim'],
            num_layers = configs['num_layers'],
            batch_first = True
        )
        # decoder
        self.decoder = nn.GRU(
            input_size = configs['feature_dim'],
            hidden_size = configs['hidden_dim'],
            num_layers = configs['num_layers'],
            batch_first = True
        )
        # hidden dimension --> 단변량 예측
        self.proj = nn.Linear(configs['hidden_dim'], configs['feature_dim'])

    def forward(self, x):
        _, h = self.encoder(x) # [B, lookback, hidden_dim], [num_layers, B, hidden_dim]

        # decoder 첫 번째 input = input sequence의 마지막 관측값
        dec_in = x[:, -1:, :] # [B, 1, feature_dim]

        outputs = []
        for t in range(self.configs['horizon']):
            dec_out, h = self.decoder(dec_in, h) # [B, 1, hidden_dim], [1, B, hidden_dim]
            y_hat = self.proj(dec_out) # [B, 1, feature_dim]
            outputs.append(y_hat)
            # 다음 입력을 업데이트
            dec_in = y_hat

        return torch.cat(outputs, dim=1) # [B, horizon, feature_dim]

model = Forecaster(configs)
model.train()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 가능한 파라미터 수: {total_params}")
#%%
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
#%%
### training
train_history = []
val_history   = []
for epoch in range(configs['epochs']):
    model.train()
    train_losses = []

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        pred = model(x_batch) # [B, horizon, feature_dim]

        loss = loss_fn(pred, y_batch) # MSE
        train_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

    train_mse = float(np.mean(train_losses))

    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            pred = model(x_batch) # [B, horizon, feature_dim]
            loss = loss_fn(pred, y_batch) # MSE
            val_losses.append(loss.item())
    val_mse = float(np.mean(val_losses))

    train_history.append(train_mse)
    val_history.append(val_mse)
    print(f"Epoch {epoch+1:3d}/{configs['epochs']} | Train MSE {train_mse:.4f} | Val MSE {val_mse:.4f}")
#%%
### loss values 시각화
plt.figure(figsize=(7, 3))
plt.plot(train_history, linewidth=2, label="Train")
plt.plot(val_history,  linewidth=2, label="Val")
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("MSE", fontsize=13)
plt.title("MSE 손실함수 곡선")
plt.grid(alpha=0.3)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig("./fig/loss_curve.png")
plt.show()
plt.close()
#%%
### test dataset forecasting
model.eval()
forecast = []
true = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = model(x_batch) # [B, horizon, feature_dim]
        forecast.append(pred)
        true.append(y_batch)
forecast = torch.cat(forecast, dim=0).squeeze()
true = torch.cat(true, dim=0).squeeze()

print(forecast.shape)
print(true.shape)
#%%
test_mse = (forecast - true).pow(2).mean()
print(f"Test dataset MSE: {test_mse:.4f}")
#%%
### horizon 간격으로 시계열을 이동
forecast = forecast[::configs['horizon'], :].flatten()
true = true[::configs['horizon'], :].flatten()
#%%
# 원본 scale로 변환
forecast_scaled = forecast * std + mean
true_scaled = true * std + mean
#%%
plt.figure(figsize=(10, 4))
plt.plot(forecast_scaled, linewidth=1, label="Forecasting")
plt.plot(true_scaled, linewidth=1, label="Test")
plt.xlabel("timestep", fontsize=14)
plt.ylabel("T (degC)", fontsize=14)
plt.title(f"Test dataset 예측 결과 (horizon={configs['horizon']})", fontsize=15)
plt.grid(alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("./fig/forecast.png")
plt.show()
plt.close()
#%%