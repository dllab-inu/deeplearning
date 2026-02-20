#%%
import torch
import torch.nn as nn
torch.manual_seed(42)
#%%
### 1D convolution
conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
x = torch.randn(size=(1, 1, 4)) # [batch_size, in_channels, D]
output = conv(x)

assert torch.equal(output[0, 0, 0], (x[0, :, [0, 1]] * conv.weight[0, ]).sum())
assert torch.equal(output[0, 0, 1], (x[0, :, [1, 2]] * conv.weight[0, ]).sum())
assert torch.equal(output[0, 0, 2], (x[0, :, [2, 3]] * conv.weight[0, ]).sum())

print(conv.weight.shape) # [out_channels, in_channels, kernel_size]
print(output.shape) # [batch_size, out_channels, D']
#%%
### 2D convolution
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
x = torch.randn(size=(1, 1, 3, 4)) # [batch_size, in_channels, H, W]
output = conv(x)

print(conv.weight.shape) # [out_channels, in_channels, kernel_size, kernel_size]
print(output.shape) # [batch_size, out_channels, H', W']
#%%
### 1D zero padding
conv = nn.Conv1d(
    in_channels=1, out_channels=1, kernel_size=2, 
    padding='same', padding_mode='zeros')
x = torch.randn(size=(1, 1, 4)) # [batch_size, in_channels, D]
output = conv(x)

assert x.shape == output.shape
print(output.shape) # [batch_size, out_channels, D]
#%%
### 2D zero padding
conv = nn.Conv2d(
    in_channels=1, out_channels=1, kernel_size=2, 
    padding='same', padding_mode='zeros')
x = torch.randn(size=(1, 1, 3, 4)) # [batch_size, in_channels, H, W]
output = conv(x)

assert x.shape == output.shape
print(output.shape) # [batch_size, out_channels, H, W]
#%%
### strided convolution
conv = nn.Conv2d(
    in_channels=1, out_channels=1, kernel_size=2, 
    stride=2)
x = torch.randn(size=(1, 1, 4, 4))
output = conv(x)

print(output.shape)
#%%
### 1D convolution with Multiple channels
conv = nn.Conv1d(
    in_channels=5, out_channels=1, kernel_size=2)
x = torch.randn(size=(1, 5, 4)) # [batch_size, in_channels, D]
output = conv(x)

print(conv.weight.shape) # [out_channels, in_channels, kernel_size]
print(output.shape) # [batch_size, out_channels, D']
#%%
### 2D convolution with Multiple channels
conv = nn.Conv2d(
    in_channels=5, out_channels=1, kernel_size=2)
x = torch.randn(size=(1, 5, 3, 4)) # [batch_size, in_channels, H, W]
output = conv(x)

print(conv.weight.shape) # [out_channels, in_channels, kernel_size, kernel_size]
print(output.shape) # [batch_size, out_channels, H', W']
#%%
### 1D convolution with multiple outputs
conv = nn.Conv1d(
    in_channels=5, out_channels=16, kernel_size=2)
x = torch.randn(size=(1, 5, 4)) # [batch_size, in_channels, D]
output = conv(x)

print(conv.weight.shape) # [out_channels, in_channels, kernel_size]
print(output.shape) # [batch_size, out_channels, D']
#%%
### 2D convolution with multiple outputs
conv = nn.Conv2d(
    in_channels=5, out_channels=16, kernel_size=2)
x = torch.randn(size=(1, 5, 3, 4)) # [batch_size, in_channels, H, W]
output = conv(x)

print(conv.weight.shape) # [out_channels, in_channels, kernel_size, kernel_size]
print(output.shape) # [batch_size, out_channels, H', W']
#%%
### 1D 1x1 convolution
conv = nn.Conv1d(
    in_channels=5, out_channels=16, kernel_size=1)
x = torch.randn(size=(1, 5, 4)) # [batch_size, in_channels, D]
output = conv(x)

print(conv.weight.shape) # [out_channels, in_channels, kernel_size]
print(output.shape) # [batch_size, out_channels, D]
#%%
### 2D 1x1 convolution
conv = nn.Conv2d(
    in_channels=5, out_channels=16, kernel_size=1)
x = torch.randn(size=(1, 5, 3, 4)) # [batch_size, in_channels, H, W]
output = conv(x)

print(conv.weight.shape) # [out_channels, in_channels, kernel_size, kernel_size]
print(output.shape) # [batch_size, out_channels, H, W]
#%%