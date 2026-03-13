#%%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
### hyperparameters
configs = {
    # 1. dataset
    'num_features': 7, # input/output sequence의 변수 개수
    'lookback': 96, # lookback window
    'horizon': 24, # forecast horizon
    # 2. model
    'd_model': 128,
    'num_heads': 4,
    'dropout': 0.1,
    # 3. training
    'batch_size': 8,
}
configs['d_ff'] = 4 * configs['d_model']
assert configs['d_model'] % configs['num_heads'] == 0
configs['head_dim'] = configs['d_model'] // configs['num_heads']
#%%
### dummy data
# input sequence
src = torch.randn(
    configs['batch_size'], configs['lookback'], configs['num_features'], device=device)
# output sequence
tgt = torch.randn(
    configs['batch_size'], configs['horizon'], configs['num_features'], device=device)
# teacher forcing을 위한 decoder input
tgt_input = torch.cat(
    [torch.zeros((tgt.size(0), 1, tgt.size(2))), tgt[:, :-1, :]], dim=1
)
tgt_label = tgt
#%%
### Encoder
# 1. input projection 
src_proj = nn.Linear(configs['num_features'], configs['d_model']).to(device)

# 2. positional encoding
src_pos_emb = nn.Embedding(configs['lookback'], configs['d_model']).to(device)
src_dropout = nn.Dropout(configs['dropout']).to(device)

# 3. self-attention
enc_Wq = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
enc_Wk = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
enc_Wv = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
enc_Wo = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)

# 4. feed-forward network and layer normalizations
enc_ln1 = nn.LayerNorm(configs['d_model']).to(device)
enc_fc1 = nn.Linear(configs['d_model'], configs['d_ff']).to(device)
enc_ln2 = nn.LayerNorm(configs['d_model']).to(device)
enc_fc2 = nn.Linear(configs['d_ff'], configs['d_model']).to(device)
#%%
### Encoder forward process
# B: batch_size
# L: lookback

# 1. projection and positional encoding
enc_x = src_proj(src) # [B, L, d_model]
src_pos_ids = torch.arange(configs['lookback'], device=device)
src_pos = src_pos_emb(src_pos_ids).unsqueeze(0) # [1, L, d_model]
enc_x = enc_x + src_pos # [B, L, d_model]
enc_x = src_dropout(enc_x) # [B, L, d_model]

# 2. self-attention
Q = enc_Wq(enc_x) # [B, L, d_model]
K = enc_Wk(enc_x) # [B, L, d_model]
V = enc_Wv(enc_x) # [B, L, d_model]

# [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
Q = Q.view(Q.size(0), Q.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
K = K.view(K.size(0), K.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
V = V.view(V.size(0), V.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)

scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(configs['head_dim']) # [B, num_heads, L, L]
attn = F.softmax(scores, dim=-1)
context = torch.matmul(attn, V) # [B, num_heads, L, head_dim]
context = context.transpose(1, 2).contiguous() # [B, L, num_heads, head_dim]
context = context.view(context.size(0), context.size(1), configs['d_model']) # [B, L, d_model]

enc_attn_out = enc_Wo(context) # [B, L, d_model]
enc_attn_out = src_dropout(enc_attn_out) # [B, L, d_model]
enc_x = enc_ln1(enc_x + enc_attn_out) # residual connection, [B, L, d_model]

# 3. feed-forward
ff = enc_fc1(enc_x) # [B, L, d_ff]
ff = F.relu(ff)
ff = src_dropout(ff)
ff = enc_fc2(ff) # [B, L, d_model]
ff = src_dropout(ff)
enc_out = enc_ln2(enc_x + ff) # [B, L, d_model]
#%%
### Decoder
# 1. input projection 
tgt_proj = nn.Linear(configs['num_features'], configs['d_model']).to(device)

# 2. positional encoding
tgt_pos_emb = nn.Embedding(configs['horizon'], configs['d_model']).to(device)
tgt_dropout = nn.Dropout(configs['dropout']).to(device)

# 3. self-attention
dec_self_Wq = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
dec_self_Wk = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
dec_self_Wv = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
dec_self_Wo = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)

# 4. cross-attention
dec_cross_Wq = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
dec_cross_Wk = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
dec_cross_Wv = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)
dec_cross_Wo = nn.Linear(configs['d_model'], configs['d_model'], bias=False).to(device)

# 5. feed-forward networks and layer normalization
dec_ln1 = nn.LayerNorm(configs['d_model']).to(device)
dec_fc1 = nn.Linear(configs['d_model'], configs['d_ff']).to(device)
dec_ln2 = nn.LayerNorm(configs['d_model']).to(device)
dec_fc2 = nn.Linear(configs['d_ff'], configs['d_model']).to(device)
dec_ln3 = nn.LayerNorm(configs['d_model']).to(device)

out_proj = nn.Linear(configs['d_model'], configs['num_features']).to(device)
#%%
### Decoder forward process
# B: batch_size
# H: horizon

# 1. projection and positional encoding
dec_x = tgt_proj(tgt_input) # [B, H, d_model]
tgt_pos_ids = torch.arange(configs['horizon'], device=device)
tgt_pos = tgt_pos_emb(tgt_pos_ids).unsqueeze(0) # [1, H, d_model]
dec_x = dec_x + tgt_pos
dec_x = tgt_dropout(dec_x)

# 2. causal mask (미래 시점 정보를 활용하는 것을 방지)
causal_mask = torch.triu(
    torch.ones(configs['horizon'], configs['horizon'], dtype=torch.bool, device=device), 
    diagonal=1
).unsqueeze(0).unsqueeze(0) # [1, 1, H, H]

# 3. self-attention
Q = dec_self_Wq(dec_x) # [B, H, d_model]
K = dec_self_Wk(dec_x) # [B, H, d_model]
V = dec_self_Wv(dec_x) # [B, H, d_model]

# [B, H, num_heads, head_dim] -> [B, num_heads, H, head_dim]
Q = Q.view(Q.size(0), Q.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
K = K.view(K.size(0), K.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
V = V.view(V.size(0), V.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)

scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(configs['head_dim']) # [B, num_heads, H, H]
scores = scores.masked_fill(causal_mask, float("-inf")) # [B, num_heads, H, H]
attn = F.softmax(scores, dim=-1)
context = torch.matmul(attn, V) # [B, num_heads, H, head_dim]
context = context.transpose(1, 2).contiguous() # [B, H, num_heads, head_dim]
context = context.view(context.size(0), configs['horizon'], configs['d_model']) # [B, H, d_model]

dec_self_out = dec_self_Wo(context) # [B, H, d_model]
dec_self_out = tgt_dropout(dec_self_out)
dec_x = dec_ln1(dec_x + dec_self_out) # residual connection, [B, H, d_model]

# 4. cross-attention
Q = dec_cross_Wq(dec_x) # [B, H, d_model]
K = dec_cross_Wk(enc_out) # [B, L, d_model]
V = dec_cross_Wv(enc_out) # [B, L, d_model]

# [B, length, num_heads, head_dim] -> [B, num_heads, length, head_dim]
Q = Q.view(Q.size(0), Q.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
K = K.view(K.size(0), K.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)
V = V.view(V.size(0), V.size(1), configs['num_heads'], configs['head_dim']).transpose(1, 2)

scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(configs['head_dim']) # [B, num_heads, H, L]
attn = F.softmax(scores, dim=-1)
context = torch.matmul(attn, V) # [B, num_heads, H, head_dim]
context = context.transpose(1, 2).contiguous() # [B, H, num_heads, head_dim]
context = context.view(context.size(0), context.size(1), configs['d_model']) # [B, H, d_model]

dec_cross_out = dec_cross_Wo(context) # [B, H, d_model]
dec_cross_out = tgt_dropout(dec_cross_out) # [B, H, d_model]
dec_x = dec_ln2(dec_x + dec_cross_out) # residual connection, [B, H, d_model]

# 5. feed-forward network
ff = dec_fc1(dec_x) # [B, H, d_ff]
ff = F.relu(ff)
ff = tgt_dropout(ff) # [B, H, d_model]
ff = dec_fc2(ff)
ff = tgt_dropout(ff)
dec_out = dec_ln3(dec_x + ff) # [B, H, d_model]

# 6. prediction
preds = out_proj(dec_out) # [B, H, num_features]
#%%