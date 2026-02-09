#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
#%%
# MSE Loss (regression)
B, d = 16, 3 # batch_size, input dimension
pred = torch.randn(B, d, dtype=torch.float32) # model output
target = torch.randn(B, d, dtype=torch.float32) # target

mse = nn.MSELoss()
loss_mse = mse(pred, target)
assert loss_mse == (pred - target).pow(2).mean()
print(loss_mse)
#%%
# MAE Loss (regression)
B, d = 16, 3 # batch_size, input dimension
pred = torch.randn(B, d, dtype=torch.float32) # model output
target = torch.randn(B, d, dtype=torch.float32) # target

mae = nn.L1Loss()
loss_mae = mae(pred, target)
assert loss_mae == (pred - target).abs().mean()
print(loss_mae)
#%%
# BCEWithLogitsLoss (Binary Classification)
logits = torch.randn(B, 1, dtype=torch.float32) # [B, 1] 차원의 logit 벡터
target = torch.randint(0, 2, (B, 1)).float() # [B, 1] 차원의 정답 (0과 1)

bce = nn.BCEWithLogitsLoss()
loss_bce = bce(logits, target)
assert loss_bce == - (target * logits.sigmoid().log() \
                   + (1 - target) * (1 - logits.sigmoid()).log()).mean()
print(loss_bce)
#%%
# CrossEntropyLoss (Multi-class Classification)
K = 5
logits = torch.randn(B, K, dtype=torch.float32) # [B, K] 차원의 logit 행렬
target = torch.randint(0, K, (B,), dtype=torch.long) # 정답 category 번호 (long tensor)

ce = nn.CrossEntropyLoss()
loss_ce = ce(logits, target)
onehot_target = F.one_hot(target, num_classes=K).float()
assert loss_ce == -(onehot_target * logits.softmax(dim=1).log()).sum(dim=1).mean()
print(loss_ce)
#%%