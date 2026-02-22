#%%
import torch
import torch.nn as nn
#%%
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # input: [B, 1, 32, 32]
        self.nets = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), # [B, 6, 28, 28]
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # [B, 6, 14, 14]

            nn.Conv2d(6, 16, kernel_size=5), # [B, 16, 10, 10]
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), # [B, 16, 5, 5]

            nn.Conv2d(16, 120, kernel_size=5), # [B, 120, 1, 1]
            nn.Tanh(),

            nn.Flatten(), # [B, 120]
            nn.Linear(120, 84), # [B, 84]
            nn.Tanh(),
            nn.Linear(84, num_classes) # [B, 10]
        )

    def forward(self, x):
        output = self.nets(x)
        return output
#%%
model = LeNet5()
x = torch.randn(8, 1, 32, 32) # [B, C, H, W]
pred = model(x)
print(pred.shape) # [B, num_classes]
#%%
x = torch.randn(8, 1, 32, 32) # [B, C, H, W]
print(f"""[{0}] Input
    - {x.shape}""")
for i, layer in enumerate(model.nets):
    x = layer(x)
    print(f"""[{i+1}] {layer}
    - {x.shape}""")
#%%
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, 
            kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels, 
            kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x # skip connection [B, C, H, W]

        out = self.conv1(x) # [B, C, H, W]
        out = self.bn1(out) # [B, C, H, W]
        out = self.relu(out) # [B, C, H, W]

        out = self.conv2(out) # [B, C, H, W]
        out = self.bn2(out) # [B, C, H, W]

        out = out + identity # residual connection [B, C, H, W]
        out = self.relu(out) # [B, C, H, W]

        return out
#%%
channels = 64
x = torch.randn(1, channels, 32, 32)
block = ResidualBlock(channels=64)
y = block(x)
print(y.shape)
assert x.shape == y.shape
#%%
class ResidualBlock2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.projection = nn.Conv2d( # 1x1 conv
            in_ch, out_ch, 
            kernel_size=1, bias=False)
        
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 
            kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 
            kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.projection(x) # skip connection [B, C', H, W]

        out = self.conv1(x) # [B, C', H, W]
        out = self.bn1(out) # [B, C', H, W]
        out = self.relu(out) # [B, C', H, W]

        out = self.conv2(out) # [B, C', H, W]
        out = self.bn2(out) # [B, C', H, W]

        out = out + identity # residual connection [B, C', H, W]
        out = self.relu(out) # [B, C', H, W]

        return out
#%%
in_ch = 16; out_ch = 64
x = torch.randn(1, in_ch, 32, 32)
block = ResidualBlock2(in_ch, out_ch)
y = block(x)
print(x.shape)
print(y.shape)
#%%