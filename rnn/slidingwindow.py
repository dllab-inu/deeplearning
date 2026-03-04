#%%
import torch
import torch.nn as nn
#%%
T = 10
L = 3; H = 2
train_len = 7

x = torch.randn(size=(T, ))
#%%
train = x[:train_len]
train_window = train.unfold(
    dimension=0, size=L + H, step=1
)
train_x = train_window[:, :L]
train_y = train_window[:, L:]

test = x[train_len - L:]
test_window = test.unfold(
    dimension=0, size=L + H, step=1
)
test_x = test_window[:, :L]
test_y = test_window[:, L:]
#%%
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
assert train_x.shape == (train_len - L - H + 1, L)
assert train_y.shape == (train_len - L - H + 1, H)
assert test_x.shape == (T - train_len - H + 1, L)
assert test_y.shape == (T - train_len - H + 1, H)
#%%