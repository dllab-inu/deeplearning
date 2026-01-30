#%%
import torch
from torch import nn # 신경망모형에 필요한 클래스와 함수

torch.manual_seed(42) # 재현성을 위한 seed 고정
#%%
p = 10
d = 5

x = torch.randn(size=(1, p)) # 행벡터(tensor)임에 주의!
print(x.shape)
#%%
linear1 = nn.Linear(in_features=p, out_features=d)
print(linear1.weight.shape)
print(linear1.bias.shape)
#%%
z = linear1(x)
print(z.shape)

z_tmp = torch.matmul(x, linear1.weight.t()) + linear1.bias
print(torch.equal(z, z_tmp)) # 두 tensor의 동일성 확인
#%%
sigmoid = nn.Sigmoid()
z = sigmoid(z)
print(z)
#%%
relu = nn.ReLU()
z = relu(z)
print(z)
#%%
linear2 = nn.Linear(in_features=d, out_features=1)
print(linear2.weight.shape)
print(linear2.bias.shape)
#%%
out = linear2(z)
print(out.shape)

out_tmp = torch.matmul(z, linear2.weight.t()) + linear2.bias
print(torch.equal(out, out_tmp)) # 두 tensor의 동일성 확인
#%%
d1 = 8; d2 = 6; d3 = 4; d4 = 2
model = nn.Sequential(
    nn.Linear(in_features=p, out_features=d1),
    nn.ReLU(),
    nn.Linear(in_features=d1, out_features=d2),
    nn.ReLU(),
    nn.Linear(in_features=d2, out_features=d3),
    nn.ReLU(),
    nn.Linear(in_features=d3, out_features=d4),
    nn.ReLU(),
    nn.Linear(in_features=d4, out_features=1)
)
print(model)
#%%
list(model.parameters())
#%%