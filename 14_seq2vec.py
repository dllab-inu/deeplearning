#%%
import torch
import torch.nn as nn
#%%
batch_size = 4
feature_dim = 10 # D
hidden_dim = 32 # d
seq_len = 6 # T
out_dim = 5 # K
#%%
rnn = nn.RNN(
    input_size=feature_dim, hidden_size=hidden_dim,
    batch_first=True
)
fc = nn.Linear(hidden_dim, out_dim)
#%%
x = torch.randn(batch_size, seq_len, feature_dim) # input sequence, [B, T, D]
out, h_T = rnn(x)
print("RNN output:", out.shape) # [B, T, d]
print("Last hidden state:", h_T.shape) # [1, B, d]
#%%
context = h_T.squeeze(0)
pred = fc(context)
print("prediction:", pred.shape)
#%%
### Bi-directional RNN
birnn = nn.RNN(
    input_size=feature_dim, hidden_size=hidden_dim,
    batch_first=True,
    bidirectional=True
)
#%%
x = torch.randn(batch_size, seq_len, feature_dim) # input sequence, [B, T, D]
out, h_T = birnn(x)
print("RNN output:", out.shape) # [B, T, 2*d]
print("Last hidden state:", h_T.shape) # [2, B, d]
#%%
fc = nn.Linear(hidden_dim * 2, out_dim)
h_forward = h_T[0] # [B, d]
h_backward = h_T[1] # [B, d]

context = torch.cat([h_forward, h_backward], dim=1) # [B, 2d]
pred = fc(context)
print("prediction:", pred.shape)
#%%