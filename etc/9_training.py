# =========================
# Adult + Embedding MLP (Complete, no collate_fn)
# + Add: train/val/test split + validation monitoring + best-model restore
# =========================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


ADULT_COLS = [
    "age","workclass","fnlwgt","education","education-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week","native-country","income"
]

# ✅ Adult에서 "진짜" 수치형 컬럼(고정)
NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def load_adult(train_path="./data/adult.data", test_path="./data/adult.test"):
    train_df = pd.read_csv(train_path, header=None, names=ADULT_COLS, skipinitialspace=True)
    test_df  = pd.read_csv(test_path,  header=None, names=ADULT_COLS, skipinitialspace=True, comment="|")

    # test 라벨의 '.' 제거
    test_df["income"] = test_df["income"].astype(str).str.replace(".", "", regex=False)

    # 합치기
    df = pd.concat([train_df, test_df], ignore_index=True)

    # 문자열 컬럼 앞뒤 공백 제거
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    # 결측 처리: '?' -> NaN -> 제거
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)

    # 타겟 만들기
    y = (df["income"] == ">50K").astype(np.int64).to_numpy()
    X = df.drop(columns=["income"])
    return X, y


class TabularEmbeddingEncoder:
    def __init__(self, cat_cols, num_cols, manual_emb_dims=None):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.manual_emb_dims = manual_emb_dims or {}

        self.cat_maps = {}
        self.cat_sizes = {}
        self.emb_dims = {}

        self.num_mean = None
        self.num_std = None

    def fit(self, X_train: pd.DataFrame):
        # categorical: value -> index
        for c in self.cat_cols:
            uniq = sorted(set(X_train[c].astype(str)))
            self.cat_maps[c] = {v: i + 1 for i, v in enumerate(uniq)}   # UNK=0, known=1..K
            self.cat_sizes[c] = 1 + len(uniq)

            # manual embedding dim 우선 적용
            if c in self.manual_emb_dims:
                self.emb_dims[c] = int(self.manual_emb_dims[c])
            else:
                # 간단 규칙(학부용)
                self.emb_dims[c] = min(32, max(2, int(round(len(uniq) ** 0.25 * 4))))

        # numeric: mean/std (train 기준)
        num = torch.tensor(X_train[self.num_cols].to_numpy(dtype=np.float32))
        self.num_mean = num.mean(0)
        self.num_std = num.std(0).clamp_min(1e-6)

    def transform(self, X: pd.DataFrame):
        # numeric -> standardized tensor
        x_num = torch.tensor(X[self.num_cols].to_numpy(dtype=np.float32))
        x_num = (x_num - self.num_mean) / self.num_std

        # categorical -> list of (N,) long tensors
        x_cat_list = []
        for c in self.cat_cols:
            idx = [self.cat_maps[c].get(v, 0) for v in X[c].astype(str)]
            x_cat_list.append(torch.tensor(idx, dtype=torch.long))

        return x_num, x_cat_list


class AdultDataset(Dataset):
    def __init__(self, x_num: torch.Tensor, x_cat_list, y: torch.Tensor):
        self.x_num = x_num              # (N, num_dim)
        self.x_cat_list = x_cat_list    # list of (N,)
        self.y = y                      # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # ✅ collate_fn 제거를 위해 categorical을 (num_cat,) 텐서로 반환
        x_cat_i = torch.stack([c[i] for c in self.x_cat_list])  # (num_cat,)
        return self.x_num[i], x_cat_i, self.y[i]


class TabularMLP(nn.Module):
    def __init__(self, num_dim, cat_sizes, emb_dims, hidden=(256, 128), dropout=0.1):
        super().__init__()

        self.emb_layers = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim)
            for vocab_size, emb_dim in zip(cat_sizes, emb_dims)
        ])

        in_dim = int(num_dim + sum(emb_dims))

        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = []
        for j, emb in enumerate(self.emb_layers):
            embs.append(emb(x_cat[:, j]))          # (B, emb_dim_j)

        x = torch.cat([x_num] + embs, dim=1)       # (B, num_dim + sum_emb_dim)
        logits = self.mlp(x).squeeze(1)            # (B,)
        return logits


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()

    probs_all = []
    y_all = []

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)

        logits = model(x_num, x_cat)
        probs = torch.sigmoid(logits).cpu().numpy()

        probs_all.append(probs)
        y_all.append(y.numpy())

    probs_all = np.concatenate(probs_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    preds = (probs_all >= 0.5).astype(int)

    acc = (preds == y_all).mean()
    auc = roc_auc_score(y_all, probs_all)
    f1  = f1_score(y_all, preds)

    return acc, auc, f1

#%%
device = "cpu"   # GPU 있으면 "cuda"

# (1) 데이터 로딩
X_df, y_np = load_adult()

# (2) 수치형/범주형 컬럼 고정
num_cols = NUM_COLS
cat_cols = [c for c in X_df.columns if c not in num_cols]

# (3) train/val/test split (예: 0.7 / 0.1 / 0.2)
set_seed(42)
n = len(X_df)
perm = torch.randperm(n).numpy()

test_size = int(0.2 * n)
val_size  = int(0.1 * n)
train_size = n - test_size - val_size

train_idx = perm[:train_size]
val_idx   = perm[train_size:train_size + val_size]
test_idx  = perm[train_size + val_size:]

X_train = X_df.iloc[train_idx].reset_index(drop=True)
X_val   = X_df.iloc[val_idx].reset_index(drop=True)
X_test  = X_df.iloc[test_idx].reset_index(drop=True)

y_train = torch.tensor(y_np[train_idx], dtype=torch.long)
y_val   = torch.tensor(y_np[val_idx], dtype=torch.long)
y_test  = torch.tensor(y_np[test_idx], dtype=torch.long)

# (4) manual embedding dims (예시)
manual_dims = {
    "sex": 2,
    "race": 3,
    "education": 8
}

# (5) 인코더 fit/transform (✅ train 기준으로만 fit)
encoder = TabularEmbeddingEncoder(cat_cols, num_cols, manual_emb_dims=manual_dims)
encoder.fit(X_train)

x_num_train, x_cat_train_list = encoder.transform(X_train)
x_num_val,   x_cat_val_list   = encoder.transform(X_val)
x_num_test,  x_cat_test_list  = encoder.transform(X_test)

# (6) 모델 입력 차원 정보
cat_sizes = [encoder.cat_sizes[c] for c in cat_cols]
emb_dims  = [encoder.emb_dims[c] for c in cat_cols]
num_dim = x_num_train.size(1)

# (7) dataset/loader
train_ds = AdultDataset(x_num_train, x_cat_train_list, y_train)
val_ds   = AdultDataset(x_num_val,   x_cat_val_list,   y_val)
test_ds  = AdultDataset(x_num_test,  x_cat_test_list,  y_test)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=4096, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=4096, shuffle=False)

# (8) 모델/옵티마이저/손실
model = TabularMLP(num_dim=num_dim, cat_sizes=cat_sizes, emb_dims=emb_dims).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

# (9) 학습 곡선 + best model tracking (val AUC 기준)
train_acc_curve = []
val_acc_curve   = []
val_auc_curve   = []

best_val_auc = -1.0
best_state = None

for epoch in range(10):
    model.train()

    for x_num, x_cat, y in train_loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.float().to(device)

        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    train_acc, _, _ = evaluate(model, train_loader, device=device)
    val_acc, val_auc, val_f1 = evaluate(model, val_loader, device=device)

    train_acc_curve.append(train_acc)
    val_acc_curve.append(val_acc)
    val_auc_curve.append(val_auc)

    # ✅ best model 저장 (val_auc 기준)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"Epoch {epoch+1}: "
        f"Train Acc={train_acc:.4f} | "
        f"Val Acc={val_acc:.4f} | "
        f"Val AUC={val_auc:.4f} | "
        f"Val F1={val_f1:.4f}"
    )

# (10) best model 복원 후 test 최종 1회 평가
if best_state is not None:
    model.load_state_dict(best_state)

test_acc, test_auc, test_f1 = evaluate(model, test_loader, device=device)
print(f"\n[Best Val AUC Model] Test Acc={test_acc:.4f} | Test AUC={test_auc:.4f} | Test F1={test_f1:.4f}")

# (11) 학습 곡선 시각화
plt.figure()
plt.plot(train_acc_curve, label="Train Accuracy")
plt.plot(val_acc_curve, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(val_auc_curve, label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.show()
#%%