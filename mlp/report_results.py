#%%
import pandas as pd
import numpy as np
#%%
path = "./assets/val_results.csv"
# path = "./assets/val_results_lr_scheduling.csv"
# path = "./assets/test_results.csv"
# path = "./assets/test_results_lr_scheduling.csv"
df = pd.read_csv(path)
#%%
def mean_and_se(x):
    x = x.values
    n = len(x) # 반복 횟수
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    return mean, se, n

if 'val' in path: tag = 'val'
else: tag = 'test'
metrics = [f"{tag}_loss", f"{tag}_acc", f"{tag}_f1", f"{tag}_auc"]
for m in metrics:
    mean, se, n = mean_and_se(df[m])
    # print(f"[{m}] Mean: {mean:.4f}, SE: {se:.4f}")
    print(f"[{m}] Mean: {mean:.4f}, 1.96*SE: {1.96*se:.4f}")
#%%