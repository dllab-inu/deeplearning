#%%
"""
Reference:
[1] https://github.com/probml/pyprobml/blob/master/notebooks/book1/08/learning_rate_plot.ipynb
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
ts = np.arange(100) # iteration 번호
lr0 = 1 # initial LR
#%%
piecewise = []
gamma = 0.5
for t in ts:
    passed_num_thresholds = sum(t > np.array([25, 50, 75]))
    lr = lr0 * np.power(gamma, passed_num_thresholds)
    piecewise.append(lr)
#%%
lam = 0.1
exponential = lr0 * np.exp(-lam * ts)
#%%
alpha = 0.5
beta = 1
polynomial = lr0 * np.power(beta * ts + 1, -alpha)
#%%
fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
ax[0].plot(piecewise)
ax[0].set_title("piecewise constant")
ax[1].plot(exponential)
ax[1].set_title("exponential decay")
ax[2].plot(polynomial)
ax[2].set_title("polynomial decay")
plt.tight_layout()
plt.savefig("./fig/3_lr_schedule.png")
plt.show()
plt.close()
#%%