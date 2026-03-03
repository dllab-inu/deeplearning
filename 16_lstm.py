#%%
import torch
import torch.nn as nn
#%%
batch_size = 4
in_dim = 10 # input sequence의 dimension
out_dim = 20 # output sequence의 dimension
hidden_dim = 32
T_enc = 8 # input sequence의 길이
T_dec = 6 # output sequence의 길이
#%%
# LSTM encoder/decoder
encoder = nn.LSTM(
    input_size=in_dim, hidden_size=hidden_dim,
    batch_first=True
)
decoder = nn.LSTM(
    input_size=out_dim, hidden_size=hidden_dim,
    batch_first=True
)
fc = nn.Linear(hidden_dim, out_dim)
#%%
x = torch.randn(batch_size, T_enc, in_dim) # input sequence

enc_out, (h, c) = encoder(x) # RNN과 달리 (hidden state, cell state)이 추가로 return
print("encoder output:", enc_out.shape) # [B, T_enc, hidden_dim]
print("(context) hidden:", h.shape) # [1, B, hidden_dim]
print("(context) cell:", c.shape) # [1, B, hidden_dim]
#%%
y_t = torch.zeros(batch_size, 1, out_dim)   # decoding 시작을 알려주는 0벡터

outputs = []
for t in range(T_dec):
    dec_out, (h, c) = decoder(y_t, (h, c)) # hidden state과 cell state 2개를 모두 입력
    pred = fc(dec_out) # [B, 1, out_dim]
    outputs.append(pred)
    y_t = pred # 다음 입력을 update
#%%
y_hat = torch.cat(outputs, dim=1)
print("prediction:", y_hat.shape) # [B, T_dec, out_dim]
#%%
### Stacked LSTM
num_layers = 3 # layer의 개수

stacked_encoder = nn.LSTM(
    input_size=in_dim, hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True
)
stacked_decoder = nn.LSTM(
    input_size=out_dim, hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True
)
fc = nn.Linear(hidden_dim, out_dim)
#%%
x = torch.randn(batch_size, T_enc, in_dim)  # input sequence

enc_out, (h, c) = stacked_encoder(x)
print("encoder output:", enc_out.shape) # [B, T_enc, hidden_dim]
print("(context) hidden:", h.shape) # [num_layers, B, hidden_dim]
print("(context) cell:", c.shape) # [num_layers, B, hidden_dim]
#%%
y_t = torch.zeros(batch_size, 1, out_dim)  # decoding 시작을 알려주는 0벡터

outputs = []
for t in range(T_dec):
    dec_out, (h, c) = stacked_decoder(y_t, (h, c))
    # dec_out: [B, 1, hidden_dim]
    pred = fc(dec_out)
    outputs.append(pred)
    y_t = pred
#%%
y_hat = torch.cat(outputs, dim=1)
print("prediction:", y_hat.shape) # [B, T_dec, out_dim]
#%%