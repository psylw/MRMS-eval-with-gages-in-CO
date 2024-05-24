#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

window = pd.read_feather('../output/window_values_new')
window = window.loc[window.total_mrms_accum>1].reset_index(drop=True)
window = window.dropna()

train = pd.read_feather('../output/train_test2')
train = train.loc[train.total_mrms_accum>1].reset_index(drop=True)
train = train.dropna()


# %%
# calculate other error metrics
import sys
sys.path.append('../utils')
from mean_error import me

# nRMSE
# mean bias
window['mean_error'] = me(window)

#%%
# how do metrics vary with year
train['mean_error'] = window.mean_error
train['year'] = [train.start[i].year for i in train.index]
train['nrmse'] = (train.norm_diff/train.max_mrms).values
train['nme'] = (train.mean_error/train.max_mrms).values
#%%
train = train.loc[train.total_mrms_accum>10].reset_index(drop=True)
#%%
sns.boxplot(data = train,x='year',y='mean_error')
plt.ylim(-10,10)
plt.show()
sns.boxplot(data = train,x='year',y='norm_diff')
plt.ylim(0,10)
plt.show()
sns.boxplot(data = train,x='year',y='rqi_mean')
plt.show()
sns.boxplot(data = train,x='year',y='nrmse')
plt.ylim(0,1)
plt.show()
sns.boxplot(data = train,x='year',y='nme')
plt.ylim(-.5,.5)
plt.show()
# how do metrics vary with month
# %%
