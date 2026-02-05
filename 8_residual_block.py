#%%
import torch
import torch.nn as nn
#%%
class MLPResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        residual = x # [B, dim]
        out = self.fc1(x) # [B, hidden_dim]
        out = self.activation(out) # [B, hidden_dim]
        out = self.fc2(out) # [B, dim]
        return residual + out # [B, dim]
#%%
B = 32 # batch_size
dim = 4; hidden_dim = 8

residual_block = MLPResidualBlock(dim=dim, hidden_dim=hidden_dim)
residual_block.train()
#%%
x = torch.randn(size=(B, dim))
out = residual_block(x)
print(out.shape)
assert x.shape == out.shape
#%%