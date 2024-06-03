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
# nRMSE
# mean bias
window['mean_error'] = pd.read_feather('../output/experiments/mean_error_values')

#%%
# how do metrics vary with year
train['mean_error'] = window.mean_error
train['year'] = [train.start[i].year for i in train.index]
train['nRMSE'] = (train.norm_diff/train.max_mrms).values
train['nME'] = (train.mean_error/train.max_mrms).values

train = train.rename(columns={'norm_diff':'RMSE'})
#%%
# define lat band
q=[]
for i in range(0,5):
    q.append(train.quantile(.25*i).mrms_lon)

train['long_band'] = pd.cut(train['mrms_lon'], bins=q, labels=list(range(1, 5)))

q=[]
for i in range(0,5):
    q.append(train.quantile(.25*i).mrms_lat)

train['lat_band'] = pd.cut(train['mrms_lat'], bins=q, labels=list(range(1, 5)))

#%%
#train_10 = train.loc[train.total_mrms_accum>10].reset_index(drop=True)
#%%
sns.boxplot(data = train,x='year',y='mean_error', hue='month')
plt.ylim(-1,2)
plt.show()
sns.boxplot(data = train,x='year',y='norm_diff', hue='month')
plt.ylim(0,10)
plt.show()
sns.boxplot(data = train,x='year',y='rqi_mean', hue='month')
plt.show()
sns.boxplot(data = train,x='year',y='nrmse', hue='month')
plt.ylim(0,1)
plt.show()
sns.boxplot(data = train,x='year',y='nme', hue='month')
plt.ylim(-.5,1)
plt.legend(loc='upper right')
plt.show()
# how do metrics vary with month
# %%
