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
q_lon=[]
for i in range(0,5):
    q_lon.append(train.quantile(.25*i).mrms_lon)

train['long_band'] = pd.cut(train['mrms_lon'], include_lowest=True, bins=q_lon, labels=list(range(1, 5)))

q_lat=[]
for i in range(0,5):
    q_lat.append(train.quantile(.25*i).mrms_lat)

train['lat_band'] = pd.cut(train['mrms_lat'], include_lowest=True, bins=q_lat, labels=list(range(1, 5)))

q_e=[]
for i in range(0,5):
    q_e.append(train.quantile(.25*i).point_elev)

train['e_band'] = pd.cut(train['point_elev'], include_lowest=True,bins=q_e, labels=list(range(1, 5)))
#%%
#train_10 = train.loc[train.total_mrms_accum>10].reset_index(drop=True)
#%%
plot_name = 'S6'
data = train[train.year>2020]
x = 'month'
#error = ['mean_error','nME','RMSE','nRMSE','rqi_mean']
error = ['RMSE','nRMSE','rqi_mean']

y_label = ['RMSE','nRMSE','mean RQI']
#y_label = ['mean_error','nME','RMSE','nRMSE','rqi_mean']
hue = 'e_band'

ylim = [(0,10),(0,1),(0,1)]
#ylim = [(-1,3),(-.5,1),(0,10),(0,1),(0,1)]

#labels = [f'{round(q_lon[i]-360,2)} - {round(q_lon[i+1]-360,2)}' for i in range(len(q_lon)-1)]

#labels = [f'{round(q_lat[i],2)} - {round(q_lat[i+1],2)}' for i in range(len(q_lat)-1)]

labels = [f'{int(q_e[i])} - {int(q_e[i+1])}' for i in range(len(q_e)-1)]

fig, axs = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={'wspace': 0.3})

for idx,y in enumerate(error):
    sns.boxplot(data = data,x=x,y=y, hue=hue,ax=axs[idx])

    axs[idx].set_ylim(ylim[idx][0],ylim[idx][1])
    axs[idx].legend(loc='upper right')
    axs[idx].set_xlabel(x, fontsize=20)
    axs[idx].set_ylabel(y_label[idx], fontsize=20)
    axs[idx].tick_params(axis='both', which='major', labelsize=16)

    if idx==0:
        # Update legend labels
        handles, _ = axs[idx].get_legend_handles_labels()
        axs[idx].legend(handles, labels[:len(handles)], loc='upper right')
    else:
        axs[idx].legend().remove()


fig.savefig("../output_figures/experiments/"+plot_name+".pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')

# %%
