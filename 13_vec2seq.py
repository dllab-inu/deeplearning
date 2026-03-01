#%%
import torch
import torch.nn as nn
#%%
batch_size = 4
input_dim = 8
feature_dim = 10 # D
hidden_dim = 32 # d
seq_len = 6 # T
#%%
x = torch.randn(batch_size, input_dim)
encoder = nn.Linear(input_dim, hidden_dim)
h = encoder(x).tanh() # h0, [B, d]
print("hidden size:", h.shape)

h = h.unsqueeze(0)
print("(RNN format) hidden size:", h.shape) # [1, B, d]
#%%
rnn = nn.RNN(
    input_size=feature_dim, hidden_size=hidden_dim,
    batch_first=True
)
fc = nn.Linear(hidden_dim, feature_dim)
#%%
y_t = torch.zeros(batch_size, 1, feature_dim) # y0 (0벡터를 이용), [B, 1, D]

preds = []
for t in range(seq_len):
    out, h = rnn(y_t, h) # [B, 1, d], [1, B, d]
    pred = fc(out) # [B, 1, D]
    preds.append(pred)
    y_t = pred # 다음 입력을 update
#%%
assert seq_len == len(preds)
print("single prediction:", preds[-1].shape)
y_hat = torch.cat(preds, dim=1)
print("prediction:", y_hat.shape)
#%%