#%%
import torch
import torch.nn as nn
#%%
### 2D max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(size=(1, 10, 4, 4)) # [batch_size, in_channels, H, W]
output = max_pool(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
#%%
### 2D avg pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
x = torch.randn(size=(1, 10, 4, 4)) # [batch_size, in_channels, H, W]
output = avg_pool(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
#%%
### Global Average Pooling 
x = torch.randn(size=(1, 16, 4, 5)) # [batch_size, in_channels, H, W]
gap_layer = nn.AdaptiveAvgPool2d(output_size=1) # Global Average Pooling layer
output = gap_layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
#%%
x = torch.randn(size=(1, 16, 4, 5)) # [batch_size, in_channels, H, W]
gap_layer = nn.AdaptiveAvgPool2d(output_size=1) # Global Average Pooling layer
output = gap_layer(x)

flatten_layer = nn.Flatten()
fc_layer = nn.Linear(16, 5)
logit = fc_layer(flatten_layer(output))

flatten_layer(x).shape

print(flatten_layer(output).shape)
print(f"Logit shape: {logit.shape}")
#%%
### Batch normalization (2D input)
x = torch.randn(size=(256, 4))
batchnorm = nn.BatchNorm1d(num_features=4)
output = batchnorm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(output.mean(dim=0))
print(output.var(dim=0, correction=0))

print(batchnorm.weight)
print(batchnorm.weight.shape) # gamma
print(batchnorm.bias)
print(batchnorm.bias.shape) # beta
#%%
### (참고)
print(batchnorm.running_mean) # mu (empirical mean)
print(batchnorm.running_mean.shape)
print(x.mean(dim=0) * batchnorm.momentum)

print(batchnorm.running_var) # sigma (empirical variance)
print(batchnorm.running_var.shape)
print(x.var(dim=0, correction=1) * batchnorm.momentum + 1 * (1 - batchnorm.momentum))
#%%
### Batch normalization (3D input)
x = torch.randn(size=(256, 4, 3))
batchnorm = nn.BatchNorm1d(num_features=4)
output = batchnorm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(output.mean(dim=[0, 2]))
print(output.var(dim=[0, 2], correction=0))

print(batchnorm.weight)
print(batchnorm.weight.shape) # gamma
print(batchnorm.bias)
print(batchnorm.bias.shape) # beta
#%%
### Batch normalization (4D input)
x = torch.randn(size=(256, 4, 3, 2))
batchnorm = nn.BatchNorm2d(num_features=4)
output = batchnorm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(output.mean(dim=[0, 2, 3]))
print(output.var(dim=[0, 2, 3], correction=0))

print(batchnorm.weight)
print(batchnorm.weight.shape) # gamma
print(batchnorm.bias)
print(batchnorm.bias.shape) # beta
#%%
### Layer normalization (2D input)
x = torch.randn(size=(4, 32))
layernorm = nn.LayerNorm(normalized_shape=32)
output = layernorm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(output.mean(dim=1))
print(output.var(dim=1, correction=0))

print(layernorm.weight.shape) # gamma
print(layernorm.bias.shape) # beta
#%%
### Layer normalization (3D input)
x = torch.randn(size=(4, 32, 16))
layernorm = nn.LayerNorm(normalized_shape=[32, 16])
output = layernorm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(output.mean(dim=[1, 2]))
print(output.var(dim=[1, 2], correction=0))

print(layernorm.weight.shape) # gamma
print(layernorm.bias.shape) # beta
#%%
### Layer normalization (4D input)
x = torch.randn(size=(4, 32, 16, 8))
layernorm = nn.LayerNorm(normalized_shape=[32, 16, 8])
output = layernorm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(output.mean(dim=[1, 2, 3]))
print(output.var(dim=[1, 2, 3], correction=0))

print(layernorm.weight.shape) # gamma
print(layernorm.bias.shape) # beta
#%%