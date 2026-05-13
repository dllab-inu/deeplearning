#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#%%
configs = {
    'seed': 42,
    'neg_ratio': 4,

    'latent_dim': 32,
    'num_layers': 2,
    'dropout': 0.1,

    'epochs': 100,
    'batch_size': 256,
    'lr': 0.005,

    'n_neg_eval': 99,
    'topk': 10,
}
#%%
torch.manual_seed(configs['seed'])
np.random.seed(configs['seed'])
#%%
df = pd.read_csv("./data/movielens.csv", index_col=0)
df = df.loc[df['user_id'] < 100]

n_users = df["user_id"].nunique()
n_items = df["movie_id"].nunique()
print('n_users:', n_users)
print('n_items:', n_items)
#%%
### leave-one-out split
df_sorted = df.sort_values("timestamp")
 
# index of the last interaction per user
test_idx = df_sorted.groupby("user_id").tail(1).index

test_df = df_sorted.loc[test_idx].reset_index(drop=True)
train_df = df_sorted.drop(index=test_idx).reset_index(drop=True)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
assert test_df.groupby("user_id").size().max() == 1
assert test_df.groupby("user_id").size().min() == 1
#%%
### implicit feedback dataset with negative sampling
class NCFDataset(Dataset):
    def __init__(
        self, df, n_items, neg_ratio=4, is_train=True,
    ):
        self.n_items = n_items
        self.neg_ratio = neg_ratio
        self.is_train = is_train

        # user_pos_items[u] = set of item indices user u interacted with
        self.user_pos_items = df.groupby("user_id")["movie_id"].apply(set).to_dict()

        self.users = df["user_id"].values # [N_pos, ]
        self.items = df["movie_id"].values # [N_pos, ]
        self.labels = np.ones(len(df), dtype=np.float32) # all 1 (positive)

        if is_train:
            self._add_negatives()

    def _sample_negative(self, user):
        """sample item which is not in set of item indices user u interacted with"""
        pos = self.user_pos_items[user]
        while True:
            j = np.random.randint(0, self.n_items)
            if j not in pos:
                return j

    def _add_negatives(self):
        neg_users, neg_items = [], []
        for u in self.users:
            for _ in range(self.neg_ratio):
                neg_users.append(u)
                neg_items.append(self._sample_negative(u))

        self.users = np.concatenate([self.users, neg_users]) # [N_pos + neg_ratio * N_pos, ]
        self.items = np.concatenate([self.items, neg_items]) # [N_pos + neg_ratio * N_pos, ]
        neg_labels = np.zeros(len(neg_users), dtype=np.float32)
        self.labels = np.concatenate([self.labels, neg_labels]) # [N_pos + neg_ratio * N_pos, ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

train_dataset = NCFDataset(train_df, n_items, neg_ratio=configs['neg_ratio'], is_train=True)
train_loader  = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)

assert len(train_dataset.users) == len(train_df) * (1 + configs['neg_ratio'])
print('number of samples:', len(train_dataset.users))
#%%
class NCF(nn.Module):
    def __init__(
        self, n_users, n_items, latent_dim, num_layers, dropout,
    ):
        super().__init__()
 
        # embedding layer
        self.user_emb = nn.Embedding(n_users, latent_dim)
        self.item_emb = nn.Embedding(n_items, latent_dim)
 
        layers = []
        dim = 2 * latent_dim # first layer input: concatenated embeddings
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, latent_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim = latent_dim
        self.nets = nn.Sequential(*layers)
 
        self.output_layer = nn.Linear(latent_dim, 1)
 
    def forward(self, user_ids, item_ids):
        p_u = self.user_emb(user_ids) # [B, D] (D: latent_dim)
        q_i = self.item_emb(item_ids) # [B, D] (D: latent_dim)
 
        z = torch.cat([p_u, q_i], dim=-1) # [B, 2*D]
 
        z = self.nets(z) # [B, dim]
 
        logit = self.output_layer(z).squeeze(-1) # [B, ]
 
        return logit

model = NCF(n_users, n_items, configs['latent_dim'], configs['num_layers'], configs['dropout'])
model.train()
#%%
optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
criterion = nn.BCEWithLogitsLoss()
#%%
loss_history = []
for epoch in range(configs['epochs']):
    loss_ = []

    for user_ids, item_ids, labels in train_loader:
        optimizer.zero_grad()

        logits = model(user_ids, item_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        loss_.append(loss.item())
    loss_history.append(np.average(loss_))
    print(f"Epoch: {epoch+1:3d}/{configs['epochs']} | Train Loss: {loss_history[-1]:.4f}")
#%%
### loss values 시각화
plt.figure(figsize=(7, 3))
plt.plot(loss_history, linewidth=2, label="Train")
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("BCE", fontsize=13)
plt.title("BCE loss curve")
plt.grid(alpha=0.3)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()
plt.close()
#%%
model.eval()
train_pos = train_dataset.user_pos_items

hits = []
for _, row in test_df.iterrows():
    u = int(row["user_id"])
    pos_item = int(row["movie_id"])

    # sample n_neg_eval negatives
    neg_pool = train_pos.get(u, set())
    neg_items = []
    while len(neg_items) < configs['n_neg_eval']:
        j = np.random.randint(0, n_items)
        if j != pos_item and j not in neg_pool:
            neg_items.append(j)

    # build candidate list: [pos_item] + 99 negatives
    candidates = [pos_item] + neg_items # 100 items

    u_tensor = torch.tensor([u] * 100, dtype=torch.long)
    i_tensor = torch.tensor(candidates, dtype=torch.long)
    with torch.no_grad():
        scores = model(u_tensor, i_tensor).numpy() # [100, ]

    rank = np.argsort(-scores).tolist().index(0) + 1 # rank of positive sample

    hits.append(1 if rank <= configs['topk'] else 0)

print(f'HitRatio@K: {np.mean(hits):.4f}')
#%%