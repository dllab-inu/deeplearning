#%%
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
batch_size = 4
in_dim = 10 # input sequence의 dimension
out_dim = 20 # output sequence의 dimension
hidden_dim = 32
T_enc = 8 # input sequence의 길이 = L
T_dec = 6 # output sequence의 길이
#%%
encoder = nn.GRU(
    input_size=in_dim, 
    hidden_size=hidden_dim,
    batch_first=True
)
decoder = nn.GRU(
    input_size=out_dim+hidden_dim, # Input-feeding approach
    hidden_size=hidden_dim, 
    batch_first=True
)
fc = nn.Linear(
    hidden_dim*2, hidden_dim, bias=False
)
#%%
x = torch.randn(batch_size, T_enc, in_dim)  # input sequence

enc_out, enc_context = encoder(x)
print("encoder output:", enc_out.shape) # [B, T_enc, hidden_dim]
print("encoder context:", enc_context.shape) # [1, B, hidden_dim]
#%%
### step 1 and 2
y = torch.zeros((batch_size, 1, out_dim)) # start of sequence
h_tilde = torch.zeros((batch_size, 1, hidden_dim)) # initialize attentional vector input
h = enc_context # initialize hidden state

_, h = decoder(
    torch.cat([y, h_tilde], dim=-1), 
    h
)
print("h.shape:", h.shape) # decoder hidden, [1, B, hidden_dim]
#%%
### step 3
dot_product = torch.matmul(
    h.permute(1, 0, 2), # [B, 1, hidden_dim]
    enc_out.permute(0, 2, 1) # [B, hidden_dim, T_enc]
) / np.sqrt(hidden_dim)
print("dot_product.shape:", dot_product.shape) # [B, 1, T_enc]

attention_score = dot_product.softmax(dim=-1)
print("attention_score.shape:", attention_score.shape) # [B, 1, T_enc]
#%%
### step 4
context = torch.matmul(attention_score, enc_out)
print("context.shape:", context.shape) # [B, 1, hidden_dim]
#%%
### step 5
h_tilde = fc(
    torch.cat([h.permute(1, 0, 2), context], dim=-1) # [B, 1, hidden_dim*2]
).tanh()
print("h_tilde.shape:", h_tilde.shape) # [B, 1, hidden_dim]
#%%