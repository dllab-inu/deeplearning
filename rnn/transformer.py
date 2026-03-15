#%%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
#%%
### 데이터 불러오기 (다변량)
df = pd.read_csv("./data/climate_multi.csv", parse_dates=['Date Time'], index_col=0)
print(df.shape)
print(df.head())
#%%
configs = {
    'seed': 42,
    'num_features': 14,

    'lookback': 48,
    'horizon': 6,
    'val_ratio': 0.2,

    'd_model': 32,
    'num_heads': 2,
    'dropout': 0.1,

    'epochs': 30,
    'batch_size': 128,
    'lr': 0.001,
}
configs['d_ff'] = 4 * configs['d_model']
assert configs['d_model'] % configs['num_heads'] == 0
configs['head_dim'] = configs['d_model'] // configs['num_heads']

### Reproducibility
torch.manual_seed(configs['seed'])
np.random.seed(configs['seed'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
### train, val / test split
trainval_len = len(df.loc[df['Date Time'] < "2016-10-01"])
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
train = train.values[:, 1:].astype("float32") # 첫 번째 Date Time 열을 제거, [T, D]
train = torch.FloatTensor(train)

val = val.values[:, 1:].astype("float32") # 첫 번째 Date Time 열을 제거, [T, D]
val = torch.FloatTensor(val)

test = test.values[:, 1:].astype("float32") # 첫 번째 Date Time 열을 제거, [T, D]
test = torch.FloatTensor(test)
#%%
### 표준화 - 학습데이터의 통계량만을 이용
mean = train.mean(axis=0, keepdim=True)
std = train.std(axis=0, keepdim=True) + 1e-6

train = (train - mean) / std
val = (val - mean) / std
test = (test - mean) / std
#%%
### window format dataset
    # dimension=0 - 시간 축 방향으로 슬라이딩
    # size - 윈도우 한 개의 길이
    # step = 1 - 한 timestep씩 이동
window_len = configs['lookback'] + configs['horizon']

train = train.unfold(
    dimension=0, size=window_len, step=1
).permute(0, 2, 1) # [T, D, lookback+horizon] --> [T, lookback+horizon, D]

val = val.unfold(
    dimension=0, size=window_len, step=1
).permute(0, 2, 1)

test = test.unfold(
    dimension=0, size=window_len, step=1
).permute(0, 2, 1)

print(train.shape) # [T, lookback+horizon, D]
print(val.shape)
print(test.shape)
#%%
### torch dataset
train_dataset = TensorDataset(
    train[:, :configs['lookback'], :], # [T, lookback, D]
    train[:, configs['lookback']:, :], # [T, horizon, D]
)
val_dataset = TensorDataset(
    val[:, :configs['lookback'], :],
    val[:, configs['lookback']:, :],
)
test_dataset = TensorDataset(
    test[:, :configs['lookback'], :],
    test[:, configs['lookback']:, :],
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

print("x_batch:", x_batch.shape)  # [B, lookback, D]
print("y_batch:", y_batch.shape)  # [B, horizon, D]
#%%
class Encoder(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.configs = configs
        self.device = device

        # 1. input projection 
        self.src_proj = nn.Linear(configs['num_features'], configs['d_model']).to(device)

        # 2. positional encoding
        self.src_pos_emb = nn.Embedding(configs['lookback'], configs['d_model']).to(device)
        self.src_dropout = nn.Dropout(configs['dropout']).to(device)

        # 3. self-attention
        self.enc_Wq = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.enc_Wk = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.enc_Wv = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.enc_Wo = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.enc_ln1 = nn.LayerNorm(configs['d_model']).to(device)

        # 4. feed-forward network and layer normalizations
        self.enc_fc1 = nn.Linear(configs['d_model'], configs['d_ff']).to(device)
        self.enc_fc2 = nn.Linear(configs['d_ff'], configs['d_model']).to(device)
        self.enc_ln2 = nn.LayerNorm(configs['d_model']).to(device)

    def forward(self, src):
        # src: [B, L, num_features]
        # 1. projection and positional encoding
        enc_x = self.src_proj(src) # [B, L, d_model]
        src_pos_ids = torch.arange(configs['lookback'], device=self.device)
        src_pos = self.src_pos_emb(src_pos_ids).unsqueeze(0) # [1, L, d_model]
        enc_x = enc_x + src_pos # [B, L, d_model]
        enc_x = self.src_dropout(enc_x) # [B, L, d_model]

        # 2. self-attention
        Q = self.enc_Wq(enc_x) # [B, L, d_model]
        K = self.enc_Wk(enc_x) # [B, L, d_model]
        V = self.enc_Wv(enc_x) # [B, L, d_model]

        # [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        Q = Q.view(Q.size(0), Q.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
        K = K.view(K.size(0), K.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(configs['head_dim']) # [B, num_heads, L, L]
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V) # [B, num_heads, L, head_dim]
        context = context.transpose(1, 2).contiguous() # [B, L, num_heads, head_dim]
        context = context.view(context.size(0), context.size(1), configs['d_model']) # [B, L, d_model]

        enc_attn_out = self.enc_Wo(context) # [B, L, d_model]
        enc_attn_out = self.src_dropout(enc_attn_out) # [B, L, d_model]
        enc_x = self.enc_ln1(enc_x + enc_attn_out) # residual connection, [B, L, d_model]

        # 3. feed-forward
        ff = self.enc_fc1(enc_x) # [B, L, d_ff]
        ff = F.relu(ff)
        ff = self.src_dropout(ff)
        ff = self.enc_fc2(ff) # [B, L, d_model]
        ff = self.src_dropout(ff)
        enc_out = self.enc_ln2(enc_x + ff) # [B, L, d_model]

        return enc_out
#%%
class Decoder(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.configs = configs
        self.device = device

        # 1. input projection 
        self.tgt_proj = nn.Linear(configs['num_features'], configs['d_model']).to(device)

        # 2. positional encoding
        self.tgt_pos_emb = nn.Embedding(configs['horizon'], configs['d_model']).to(device)
        self.tgt_dropout = nn.Dropout(configs['dropout']).to(device)

        # 3. self-attention
        self.dec_self_Wq = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.dec_self_Wk = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.dec_self_Wv = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.dec_self_Wo = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.dec_ln1 = nn.LayerNorm(configs['d_model']).to(device)

        # 4. cross-attention
        self.dec_cross_Wq = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.dec_cross_Wk = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.dec_cross_Wv = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.dec_cross_Wo = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
        self.dec_ln2 = nn.LayerNorm(configs['d_model']).to(device)

        # 5. feed-forward networks and layer normalization
        self.dec_fc1 = nn.Linear(configs['d_model'], configs['d_ff']).to(device)
        self.dec_fc2 = nn.Linear(configs['d_ff'], configs['d_model']).to(device)
        self.dec_ln3 = nn.LayerNorm(configs['d_model']).to(device)

        self.out_proj = nn.Linear(configs['d_model'], configs['num_features']).to(device)
    
    def forward(self, tgt, enc_out):
        # tgt: [B, H, num_features]
        # enc_out: [B, L, d_model]
        tgt_input = torch.cat(
            [torch.zeros((tgt.size(0), 1, tgt.size(2))).to(self.device), tgt[:, :-1, :]], dim=1
        ) # teacher forcing

        # 1. projection and positional encoding
        dec_x = self.tgt_proj(tgt_input) # [B, H, d_model]
        tgt_pos_ids = torch.arange(configs['horizon'], device=self.device)
        tgt_pos = self.tgt_pos_emb(tgt_pos_ids).unsqueeze(0) # [1, H, d_model]
        dec_x = dec_x + tgt_pos
        dec_x = self.tgt_dropout(dec_x)

        # 2. causal mask (미래 시점 정보를 활용하는 것을 방지)
        causal_mask = torch.triu(
            torch.ones(configs['horizon'], configs['horizon'], dtype=torch.bool, device=self.device), 
            diagonal=1
        ).unsqueeze(0).unsqueeze(0) # [1, 1, H, H]

        # 3. self-attention
        Q = self.dec_self_Wq(dec_x) # [B, H, d_model]
        K = self.dec_self_Wk(dec_x) # [B, H, d_model]
        V = self.dec_self_Wv(dec_x) # [B, H, d_model]

        # [B, H, num_heads, head_dim] -> [B, num_heads, H, head_dim]
        Q = Q.view(Q.size(0), Q.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
        K = K.view(K.size(0), K.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(configs['head_dim']) # [B, num_heads, H, H]
        scores = scores.masked_fill(causal_mask, float("-inf")) # [B, num_heads, H, H]
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V) # [B, num_heads, H, head_dim]
        context = context.transpose(1, 2).contiguous() # [B, H, num_heads, head_dim]
        context = context.view(context.size(0), configs['horizon'], configs['d_model']) # [B, H, d_model]

        dec_self_out = self.dec_self_Wo(context) # [B, H, d_model]
        dec_self_out = self.tgt_dropout(dec_self_out)
        dec_x = self.dec_ln1(dec_x + dec_self_out) # residual connection, [B, H, d_model]

        # 4. cross-attention
        Q = self.dec_cross_Wq(dec_x) # [B, H, d_model]
        K = self.dec_cross_Wk(enc_out) # [B, L, d_model]
        V = self.dec_cross_Wv(enc_out) # [B, L, d_model]

        # [B, length, num_heads, head_dim] -> [B, num_heads, length, head_dim]
        Q = Q.view(Q.size(0), Q.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
        K = K.view(K.size(0), K.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(configs['head_dim']) # [B, num_heads, H, L]
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V) # [B, num_heads, H, head_dim]
        context = context.transpose(1, 2).contiguous() # [B, H, num_heads, head_dim]
        context = context.view(context.size(0), context.size(1), configs['d_model']) # [B, H, d_model]

        dec_cross_out = self.dec_cross_Wo(context) # [B, H, d_model]
        dec_cross_out = self.tgt_dropout(dec_cross_out) # [B, H, d_model]
        dec_x = self.dec_ln2(dec_x + dec_cross_out) # residual connection, [B, H, d_model]

        # 5. feed-forward network
        ff = self.dec_fc1(dec_x) # [B, H, d_ff]
        ff = F.relu(ff)
        ff = self.tgt_dropout(ff) # [B, H, d_model]
        ff = self.dec_fc2(ff)
        ff = self.tgt_dropout(ff)
        dec_out = self.dec_ln3(dec_x + ff) # [B, H, d_model]

        # 6. prediction
        preds = self.out_proj(dec_out) # [B, H, num_features]

        return preds
#%%
class Forecaster(nn.Module):
    def __init__(self, configs, device):
        super().__init__()
        self.configs = configs
        self.device = device

        self.encoder = Encoder(configs, device)
        self.decoder = Decoder(configs, device)

    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        preds = self.decoder(tgt, enc_out)
        return preds
    
    def forecasting(self, src):
        enc_out = self.encoder(src)
        tgt = torch.zeros((src.size(0), self.configs['horizon'], src.size(2))).to(self.device)
        for t in range(self.configs['horizon']):
            preds = self.decoder(tgt, enc_out)
            tgt[:, t, :] = preds[:, t, :]
        return preds

model = Forecaster(configs, device)
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

        pred = model(x_batch, y_batch) # [B, horizon, num_features]

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
            pred = model.forecasting(x_batch) # [B, horizon, num_features]
            loss = loss_fn(pred, y_batch) # MSE
            val_losses.append(loss.item())
    val_mse = float(np.mean(val_losses))

    train_history.append(train_mse)
    val_history.append(val_mse)
    print(f"Epoch: {epoch+1:3d}/{configs['epochs']} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")
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
plt.savefig("./fig/transformer_loss_curve.png")
plt.show()
plt.close()
#%%
### test dataset forecasting
model.eval()
forecast = []
true = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = model.forecasting(x_batch) # [B, horizon, feature_dim]
        forecast.append(pred)
        true.append(y_batch)
forecast = torch.cat(forecast, dim=0).squeeze()
true = torch.cat(true, dim=0).squeeze()

print(forecast.shape)
print(true.shape)
#%%
test_mse = (forecast - true).pow(2).mean()
test_mse_lag = (forecast[1:] - true[:-1]).pow(2).mean()
print(f"Test dataset MSE: {test_mse:.4f}")
print(f"Test dataset MSE (lag 1): {test_mse_lag:.4f}")
#%%
### horizon 간격으로 시계열을 이동
forecast = forecast[::configs['horizon'], ...].flatten(start_dim=0, end_dim=1)
true = true[::configs['horizon'], ...].flatten(start_dim=0, end_dim=1)

print(forecast.shape)
print(true.shape)
#%%
# 원본 scale로 변환
forecast_scaled = forecast * std + mean
true_scaled = true * std + mean
#%%
### 기온 예측 시각화
plt.figure(figsize=(10, 4))
plt.plot(forecast_scaled[-300:, 1], linewidth=2, label="Forecasting")
plt.plot(true_scaled[-300:, 1], linewidth=2, label="Test")
plt.xlabel("timestep", fontsize=14)
plt.ylabel("T (degC)", fontsize=14)
plt.title(f"Test dataset 예측 결과 with Transformer (horizon={configs['horizon']}, MSE={test_mse:.4f})", fontsize=15)
plt.grid(alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("./fig/transformer_forecast.png")
plt.show()
plt.close()
#%%