#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#%%
configs = {
    'seed': 42,
    'hidden_dims': [128, 32],
    'latent_dim': 2,
    'epochs': 20,
    'batch_size': 256,
    'lr': 0.001,
}

torch.manual_seed(configs['seed'])
np.random.seed(configs['seed'])
#%%
### 데이터 불러오기 - MNIST (28x28 grayscale)
transform = transforms.ToTensor() # [0, 255] --> [0.0, 1.0]

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True,  drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=configs['batch_size'], shuffle=False, drop_last=False)

x_batch, _ = next(iter(train_loader))
print("x_batch:", x_batch.shape) # [B, 1, 28, 28]
print(x_batch.max(), x_batch.min())
#%%
class MLPAutoencoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        enc_layers = []
        in_dim = 28 * 28
        for h in configs['hidden_dims']:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        enc_layers += [nn.Linear(in_dim, configs['latent_dim'])]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = configs['latent_dim']
        for h in reversed(configs['hidden_dims']):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        dec_layers += [nn.Linear(in_dim, 28 * 28), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        B = x.size(0)
        return self.encoder(x.view(B, -1))  

    def forward(self, x):
        # x: [B, 1, 28, 28]
        z = self.encode(x) # [B, latent_dim]
        x_hat = self.decoder(z).view(x.size(0), 1, 28, 28) # [B, 1, 28, 28]
        return x_hat
#%%
model = MLPAutoencoder(configs)
print(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 가능한 파라미터 수: {total_params:,}")
#%%
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
#%%
### training
train_history = []
test_history  = []

for epoch in range(configs['epochs']):
    model.train()
    train_losses = []

    for x_batch, _ in train_loader: # 두 번째 인자: label
        optimizer.zero_grad()

        x_hat = model(x_batch)
        loss  = loss_fn(x_hat, x_batch)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_mse = float(np.mean(train_losses))

    # test
    model.eval()
    test_losses = []
    for x_batch, _ in test_loader:
        with torch.no_grad():
            x_hat = model(x_batch)
        loss = loss_fn(x_hat, x_batch)
        test_losses.append(loss.item())
    test_mse = float(np.mean(test_losses))

    train_history.append(train_mse)
    test_history.append(test_mse)
    print(f"Epoch: {epoch+1:3d}/{configs['epochs']} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")
#%%
### loss curve 시각화
plt.figure(figsize=(7, 3))
plt.plot(train_history, linewidth=2, label="Train")
plt.plot(test_history,  linewidth=2, label="Test")
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("MSE", fontsize=13)
plt.grid(alpha=0.3)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig(f"./fig/loss_curve_mlp_ce.png")
plt.show()
plt.close()
#%%
### test dataset reconstruction
model.eval()
x_sample, _ = next(iter(test_loader))
x_sample = x_sample[:10]

with torch.no_grad():
    x_recon = model(x_sample) # [10, 1, 28, 28]

fig, axes = plt.subplots(2, 10, figsize=(14, 3))
for i in range(10):
    axes[0, i].imshow(x_sample[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(x_recon[i].squeeze(),  cmap='gray')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(f"./fig/reconstruction_mlp_ce.png")
plt.show()
plt.close()
#%%
z_list, label_list = [], []
for x_batch, y_batch in train_loader:
    with torch.no_grad():
        z = model.encode(x_batch)
        z_list.append(z)
        label_list.append(y_batch)
z_list = torch.cat(z_list, dim=0)
label_list = torch.cat(label_list, dim=0).numpy()
#%%
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(z_list[:, 0], z_list[:, 1], c=label_list, cmap='tab10', s=3, alpha=0.5)
plt.colorbar(sc, ax=ax, ticks=range(10))
ax.set_xlabel("z1", fontsize=13)
ax.set_ylabel("z2", fontsize=13)
plt.tight_layout()
plt.savefig(f"./fig/latent_scatter_mlp_ce.png")
plt.show()
plt.close()
#%%
n_grid = 15
 
z1_min, z1_max = z_list[:, 0].min(), z_list[:, 0].max()
z2_min, z2_max = z_list[:, 1].min(), z_list[:, 1].max()
 
z1_vals = np.linspace(z1_min, z1_max, n_grid)
z2_vals = np.linspace(z2_max, z2_min, n_grid) # max --> min

model.eval()
fig, axes = plt.subplots(n_grid, n_grid, figsize=(n_grid, n_grid))
with torch.no_grad():
    for row, z2_val in enumerate(z2_vals):
        for col, z1_val in enumerate(z1_vals):
            z_pt = torch.FloatTensor([[z1_val, z2_val]]) # [1, 2]
            img = model.decoder(z_pt).view(28, 28).numpy()
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
 
plt.tight_layout()
plt.savefig(f"./fig/latent_manifold_mlp_ce.png", bbox_inches='tight')
plt.show()
plt.close()
#%%