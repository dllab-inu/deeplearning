#%%
import torch

torch.manual_seed(42) # 재현성을 위한 seed 고정
#%%
p = 10
d = 5

x = torch.randn(size=(1, p)) # 행벡터(tensor)임에 주의!
print(x.shape)
#%%
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def relu(x):
    return torch.maximum(x, torch.tensor(0))
#%%
W1 = torch.randn(size=(d, p))
b1 = torch.randn(size=(1, d))
w2 = torch.randn(size=(1, d))
b2 = torch.randn(size=(1, 1))
#%%
z = torch.matmul(x, W1.t()) + b1
print(z.shape)

z = sigmoid(z)
print(z)
#%%
z = torch.matmul(x, W1.t()) + b1
print(z.shape)

z = relu(z)
print(z)
#%%
out = torch.matmul(z, w2.t()) + b2
print(out.shape)
#%%