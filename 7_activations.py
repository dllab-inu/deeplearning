#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#%%
a = torch.linspace(-5, 5, 400)

activations = [
    ("Sigmoid", nn.Sigmoid()),
    ("Tanh", nn.Tanh()),
    ("Softplus", nn.Softplus()),
    ("ReLU", nn.ReLU()),
    ("Leaky ReLU", nn.LeakyReLU(negative_slope=0.1)),
    ("ELU", nn.ELU(alpha=0.1)),
    ("GELU", nn.GELU()),
]
#%%
fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
axes = axes.flatten()

for ax, (name, act) in zip(axes, activations):
    y = act(a)
    ax.plot(a, y, linewidth=3)
    ax.set_title(name, fontsize=17)
    ax.set_xlabel("hidden unit", fontsize=15)
    ax.grid(True)

for ax in axes[len(activations):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("./fig/4_activations.png")
plt.show()
plt.close()
#%%