#%%
import torch
#%%
### 1. scalar and vector
x = torch.randn((3, 1))
# x = torch.randn((1, 3))
y = x + 10
# y = x * 10
print("y.shape:", y.shape)
#%%
### 2. matrix and vector
A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
]) # shape: (2, 3)

b = torch.tensor([10, 20, 30])  # shape: (3, )

C = A + b # 행별로 연산이 적용됨
print("C.shape", C.shape)
#%%
A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
]) # shape: (2, 3)

b = torch.tensor([10, 20, 30])  # shape: (3, )
b = b.unsqueeze(1) # shape: (3, 1)

C = A + b
print("C.shape", C.shape)
#%%
A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
]) # shape: (2, 3)

b = torch.tensor([10, 20, 30])  # shape: (3, )
b = b.unsqueeze(0) # shape: (1, 3)

C = A + b
print("C.shape", C.shape)
#%%
### 3. row vector + col vector
row = torch.tensor([10, 20, 30]).unsqueeze(0) # shape: (1, 3)
col = torch.tensor([1, 2, 3, 4]).unsqueeze(1) # shape: (4, 1)

result = col + row
print("result.shape", result.shape)
#%%
### 4. 3D tensor
x = torch.randn((2, 3, 4)) # shape: (2, 3, 4)
bias = torch.tensor([1, 2, 3, 4]) # shape: (4,)

y = x + bias # [2, 3, 4] + [1, 1, 4]
# y = x * bias # [2, 3, 4] + [1, 1, 4]

print("y.shape", y.shape)
#%%
x = torch.randn((2, 3, 4)) # shape: (2, 3, 4)
bias = torch.tensor([10, 20]).view(2, 1, 1) # shape: (2, 1, 1)

y = x + bias

print("y.shape", y.shape)
#%%
x = torch.randn((2, 4, 3)) # shape: (2, 4, 3)
weight = torch.tensor([1, 2, 3, 4]).view(1, 4, 1) # shape: (1, 4, 1)

y = x * weight

print("y.shape", y.shape)
#%%
### 자주 하는 실수
x = torch.randn((2, 3, 4))
y = torch.randn((2, 4))

z = x + y

print("z.shape", z.shape)
#%%
x = torch.randn((2, 3, 4))
y = torch.randn((2, 1, 4))

z = x + y

print("z.shape", z.shape)
#%%
x = torch.randn((3, 4, 5))

mean1 = x.mean(dim=2)
mean2 = x.mean(dim=2, keepdim=True)

y1 = x + mean1
y2 = x + mean2

print("y1.shape", y1.shape)
print("y2.shape", y2.shape)
#%%