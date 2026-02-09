#%%
import pandas as pd
import numpy as np
#%%
path = "./assets/test_results.csv"
# path = "./assets/test_results_lr_scheduling.csv"
df = pd.read_csv(path)
#%%
def mean_and_se(x):
    x = x.values
    n = len(x) # 반복 횟수
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    return mean, se, n

metrics = ["test_loss", "test_acc", "test_f1", "test_auc"]
for m in metrics:
    mean, se, n = mean_and_se(df[m])
    print(f"[{m}] Mean: {mean:.6f}, SE: {se:.6f}")
#%%