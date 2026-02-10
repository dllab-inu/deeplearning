#%%
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#%%
### Hyperparameter configuration
configs = {
    'seed': 0,
    'hidden_dim': [64, 16],
    'val_ratio': 0.2,
    'batch_size': 1024,
    'epochs': 200,
    'lr': 0.001,

    'factor': 0.5,
    'patience': 3,
    'threshold': 1e-4
}
#%%
### Random seed 설정
random.seed(configs['seed'])
np.random.seed(configs["seed"])
torch.manual_seed(configs['seed'])
torch.cuda.manual_seed_all(configs['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#%%
### 데이터 불러오기
train_df = pd.read_csv("./data/train.csv", index_col=0)
test_df = pd.read_csv("./data/test.csv", index_col=0)

y_train_full = train_df["income"]
X_train_full = train_df.drop(columns=["income"])
y_test = test_df["income"]
X_test = test_df.drop(columns=["income"])
#%%
### 학습/검증/테스트 데이터 분할 (by random seed)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=configs["val_ratio"], random_state=configs["seed"]
)

num_cols = list(X_train.select_dtypes(include='number').columns) # 수치형 변수
obj_cols = list(X_train.select_dtypes(include=['object', 'string']).columns) # 문자형 변수

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, obj_cols), # (이름, transformation, 지정된 열)
        ("num", StandardScaler(), num_cols) # (이름, transformation, 지정된 열)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_val_encoded = preprocessor.transform(X_val)
X_test_encoded = preprocessor.transform(X_test)

# Deep learning에서는 범주형 변수를 처리할 때 reference category를 설정할 필요가 없음
encoder = preprocessor.named_transformers_["cat"]
for col, categories in zip(obj_cols, encoder.categories_):
    print(f"{col}: {len(categories)} categories")

# 최종 데이터 구성
print("[학습데이터] Input:", X_train_encoded.shape)
print("[학습데이터] Output:", y_train.shape)
print("[검증데이터] Input:", X_val_encoded.shape)
print("[검증데이터] Output:", y_val.shape)
print("[테스트데이터] Input:", X_test_encoded.shape)
print("[테스트데이터] Output:", y_test.shape)
#%%
### dataloader 구성
class TabularDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32) # 설명변수
        self.y = torch.tensor(y, dtype=torch.float32) # 반응변수

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

train_dataset = TabularDataset(X_train_encoded, y_train.values[:, None])
val_dataset = TabularDataset(X_val_encoded, y_val.values[:, None])
test_dataset = TabularDataset(X_test_encoded, y_test.values[:, None])
# y_train.values.shape
# y_train.values[:, None].shape
# train_dataset.__getitem__(3)
# X_train_encoded[3]; y_train[3]

### input data의 차원 설정
configs["input_dim"] = X_test_encoded.shape[1] # 열의 개수

train_loader = DataLoader(
    train_dataset,
    batch_size=configs["batch_size"], shuffle=True, drop_last=False
)
# x, y = next(iter(train_loader)) # 1개의 minibatch sampling
# print(x.shape); print(y.shape)
val_loader = DataLoader(
    val_dataset,
    batch_size=configs["batch_size"], shuffle=False, drop_last=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=configs["batch_size"], shuffle=False, drop_last=False
)
#%%
### device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
#%%
### 신경망모형 정의
class MLP(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        net = []
        input_dim = configs["input_dim"]
        for hidden_dim in configs["hidden_dim"]:
            net.append(nn.Linear(input_dim, hidden_dim))
            net.append(nn.ReLU())
            input_dim = hidden_dim # 이전 layer의 output과 다음 layer의 input의 차원을 동일하게 맞춤
        net.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*net) # (주의) nn.Sequential의 입력은 list가 아님

    def forward(self, x):
        return self.net(x)

model = MLP(configs).to(device)
model.train() # 학습모드
#%%
loss_function = nn.BCEWithLogitsLoss() # model의 output이 logit일 때 사용하는 이진분류 손실함수
optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max", # AUC는 클수록 좋음
    factor=configs['factor'], # lr <- lr * factor
    patience=configs['patience'], # patience epochs 동안 개선 없으면 lr 감소
    threshold=configs['threshold'], # 개선으로 인정하는 최소 변화량
)
#%%
def evaluate(model, val_loader):
    model.eval() # 평가모드

    loss_all = []
    probs_all = []
    y_all = []
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = model(x) # [B, 1]
            probs = logits.sigmoid() # 확률값으로 변환
        loss = loss_function(logits, y)
        loss_all.append(loss.item()) # scalar
        probs_all.append(probs) # [B, 1]
        y_all.append(y) # [B, 1]
    probs_all = torch.cat(probs_all).squeeze().cpu().numpy() # [n, 1] --> [n, ]
    y_all = torch.cat(y_all).squeeze().cpu().numpy() # [n, 1] --> [n, ]

    preds = (probs_all >= 0.5).astype(int)
    acc = (y_all == preds).mean()
    f1  = f1_score(y_all, preds)
    auc = roc_auc_score(y_all, probs_all)
    return np.mean(loss_all), acc, f1, auc
#%%
curve_lr = []
curve_train_loss, curve_train_acc, curve_train_f1, curve_train_auc = [], [], [], []
curve_val_loss, curve_val_acc, curve_val_f1, curve_val_auc = [], [], [], []
curve_test_loss, curve_test_acc, curve_test_f1, curve_test_auc = [], [], [], []

for epoch in range(configs["epochs"]):
    model.train()

    loss_per_epoch = []
    for x, y in train_loader:
        x = x.to(device) # minibatch를 device로 이동
        y = y.to(device) # minibatch를 device로 이동
        
        optimizer.zero_grad()
        
        logits = model(x)
        loss = loss_function(logits, y)
        loss_per_epoch.append(loss.item())

        loss.backward()
        optimizer.step()

    # 학습/검증 데이터에 대해서 평가 지표 계산
    # 실제로는, 매 epoch마다 평가를 하는 것은 많은 비용이 듦
    train_loss, train_acc, train_f1, train_auc = evaluate(model, train_loader) # 반드시 할 필요는 없음
    curve_train_loss.append(train_loss)
    curve_train_acc.append(train_acc)
    curve_train_f1.append(train_f1)
    curve_train_auc.append(train_auc)
    val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader)
    curve_val_loss.append(val_loss)
    curve_val_acc.append(val_acc)
    curve_val_f1.append(val_f1)
    curve_val_auc.append(val_auc)
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader)
    curve_test_loss.append(test_loss)
    curve_test_acc.append(test_acc)
    curve_test_f1.append(test_f1)
    curve_test_auc.append(test_auc)

    scheduler.step(val_auc) # lr scheduling step
    current_lr = optimizer.param_groups[0]["lr"] # 현재 lr
    curve_lr.append(current_lr)

    print(f"Epoch: {epoch+1:02d} | Loss: {np.mean(loss_per_epoch):.4f} | lr={current_lr:.6f}")
    print(f"--> [Validation] Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC:{val_auc:.4f}\n")
#%%
### 학습 진단 - 손실함수 및 lr
fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
axes[0].plot(curve_train_loss, linewidth=2, label="[Train] Loss")
axes[0].plot(curve_val_loss, linewidth=2, label="[Val] Loss")
axes[0].plot(curve_test_loss, linewidth=2, label="[Test] Loss")
axes[0].set_ylabel("BCE Loss", fontsize=12)
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=11)

axes[1].plot(curve_lr, linewidth=2, linestyle="--", color="black")
axes[1].set_ylabel("Learning Rate", fontsize=12)
axes[1].set_xlabel("Epochs", fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./fig/4_training_loss_lr_scheduling.png")
plt.show()
plt.close()
#%%
### 학습 진단 - 평가 지표
fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True)
axes[0].plot(curve_train_acc, linewidth=2, label='[Train] Acc')
axes[0].plot(curve_val_acc, linewidth=2, label='[Val] Acc')
axes[0].plot(curve_test_acc, linewidth=2, label='[Test] Acc')
axes[0].legend(fontsize=12)
axes[0].set_xlabel("Epochs", fontsize=12)
axes[0].set_ylabel("Accuracy", fontsize=12)
axes[0].grid(alpha=0.3)

axes[1].plot(curve_train_f1, linewidth=2, label='[Train] F1')
axes[1].plot(curve_val_f1, linewidth=2, label='[Val] F1')
axes[1].plot(curve_test_f1, linewidth=2, label='[Test] F1')
axes[1].legend(fontsize=12)
axes[1].set_xlabel("Epochs", fontsize=12)
axes[1].set_ylabel("F1", fontsize=12)
axes[1].grid(alpha=0.3)

axes[2].plot(curve_train_auc, linewidth=2, label='[Train] AUC')
axes[2].plot(curve_val_auc, linewidth=2, label='[Val] AUC')
axes[2].plot(curve_test_auc, linewidth=2, label='[Test] AUC')
axes[2].legend(fontsize=12)
axes[2].set_xlabel("Epochs", fontsize=12)
axes[2].set_ylabel("AUC-ROC", fontsize=12)
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./fig/4_metric_curve_lr_scheduling.png")
plt.show()
plt.close()
#%%
### 최종 결과
test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader)
print(f"[Test] Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC:{test_auc:.4f}\n")
#%%