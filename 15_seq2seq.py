#%%
import torch
import torch.nn as nn
#%%
batch_size = 4
in_dim = 10 # input sequence의 dimension
out_dim = 20 # output feature의 dimension
hidden_dim = 32
T_enc = 8 # input sequence의 길이
T_dec = 6 # output sequence의 길이
#%%
encoder = nn.RNN(
    input_size=in_dim, hidden_size=hidden_dim,
    batch_first=True
)

decoder = nn.RNN(
    input_size=out_dim, hidden_size=hidden_dim,
    batch_first=True
)

fc = nn.Linear(hidden_dim, out_dim)
#%%
x = torch.randn(batch_size, T_enc, in_dim) # input sequence
enc_out, h = encoder(x) # for loop이 필요 없음
print("encoder output:", enc_out.shape) # [B, T_enc, hidden_dim]
print("context:", h.shape) # context vector, [1, B, hidden_dim]
#%%
y_t = torch.zeros(batch_size, 1, out_dim) # decoding 시작을 알려주는 0벡터

outputs = []
for t in range(T_dec):
    dec_out, h = decoder(y_t, h) # encoder에서 계산된 context vector를 사용
    pred = fc(dec_out)
    outputs.append(pred)
    y_t = pred
#%%
y_hat = torch.cat(outputs, dim=1)
print("prediction:", y_hat.shape) # [B, T_dec, out_dim]
#%%
### Stacked RNN
num_layers = 3 # layer의 개수

stacked_encoder = nn.RNN(
    input_size=in_dim, hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True
)

stacked_decoder = nn.RNN(
    input_size=out_dim, hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True
)

fc = nn.Linear(hidden_dim, out_dim)
#%%
x = torch.randn(batch_size, T_enc, in_dim) # input sequence
enc_out, h = stacked_encoder(x) # for loop이 필요 없음
print("encoder output:", enc_out.shape) # [B, T_enc, hidden_dim]
print("context:", h.shape) # [num_layers, B, hidden_dim]
#%%
y_t = torch.zeros(batch_size, 1, out_dim) # decoding 시작을 알려주는 0벡터

outputs = []
for t in range(T_dec):
    dec_out, h = stacked_decoder(y_t, h) # encoder에서 계산된 hidden state 그대로 사용
    # dec_out: [B, 1, d]
    pred = fc(dec_out)
    outputs.append(pred)
    y_t = pred
#%%
y_hat = torch.cat(outputs, dim=1)
print("prediction:", y_hat.shape) # [B, T_dec, out_dim]
#%%