
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../class')

from NRMSD import nrmsd
# %%

codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)

compare = pd.read_feather('G:\PSW\working code\MRMS-eval-with-gages-in-CO\output\window_values_FN')

df = pd.read_feather(os.path.dirname(parentDir)+'\\train_test')

compare = compare.reset_index(drop=True)
compare['len_m']=[len(compare.mrms[i]) for i in range(len(compare))]
compare = compare.loc[compare.len_m>0].reset_index(drop=True)

# remove timesteps where total accum 0
compare = compare.loc[compare.total_accum_atgage>0].reset_index(drop=True)

compare=compare.rename(columns={"index": "start"})

# get rid of nan storm ids
compare.storm_id = [compare.storm_id[i][~np.isnan(compare.storm_id[i])] for i in range(len(compare))]

# remove samples where no storm_ids exist
compare['len_storm_id']=[len(compare.storm_id[i]) for i in range(len(compare))]
compare = compare.loc[compare.len_storm_id>0]

compare = compare[['mrms','gage']].reset_index(drop=True)
# %%
df = pd.concat([df,compare],axis=1).reset_index(drop=True)
# remove samples where max mrms intensity < min possible gage intensity

min_int = pd.read_feather(os.path.dirname(parentDir)+'\\min_intensity_gage')
min_int['gage_id'] = min_int.index
min_int.min_intensity = min_int.min_intensity
df['min_int'] = [min_int.loc[min_int.gage_id==df.gage_id[i][0]].min_intensity.values[0] for i in df.index]

df = df.query('max_mrms > min_int')

#df = df.reset_index(drop=True).drop(columns=['min_int','gage_id','max_accum_atgage'])
df = df.reset_index(drop=True).drop(columns=['min_int','max_accum_atgage'])
df.gage_id = [df.gage_id[i][0] for i in df.index]

# shift lon to 255.5, was 255 when i developed dataset
df = df.loc[df.longitude<255.5].dropna()



# %%
#df['norm_diff'] = np.load(os.path.join('..', 'output', 'nrmsd_mscorrect.npy'))
df['norm_diff'] = nrmsd(df)
# %%
test = df.drop(columns = ['gage_id', 'mce','mrms_accum_atgage','gage_accum', 
       'onoff', 'mrms', 'gage'])

#high = test.loc[(test.norm_diff<.05)]
#high = test.loc[(test.norm_diff<5)&(test.norm_diff>-5)]

low = test.loc[(test.norm_diff>50)|(test.norm_diff<-50)]
high = test.loc[(test.norm_diff<5)&(test.norm_diff>-5)].sample(n=len(low))
#low = df.loc[(df.mce<.5)&(df.mce>0)&(df.total_accum_atgage>10)]
#low = test.loc[(test.norm_diff>.3)].sample(n=len(high))
# %%
for i in high.columns:
    d = [high[i],low[i]]
    plt.boxplot(d)
    plt.xticks([1,2],labels=['mce>0.6','mce<0'],rotation=45)
    plt.title(i)
    plt.show()

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
high['label'] = 1
low['label'] = 0

tsne = pd.concat([high,low])

scaler = StandardScaler()

data=scaler.fit_transform(tsne.drop(columns=['label','norm_diff']))
t_sne = TSNE(n_components=2,random_state=100)
S_t_sne = t_sne.fit_transform(data)

labels = tsne.label

colors = ["navy", "darkorange","darkgreen" ]
markers=['x','+','*']

fig, axs = plt.subplots(ncols=2, nrows=2, gridspec_kw={'width_ratios': [2, 1]},figsize=(10,6))
gs = axs[0,1].get_gridspec()
# remove the underlying axes
for ax in axs[0:, 0]:
    ax.remove()
axbig = fig.add_subplot(gs[0:, 0])


for marker,color, i in zip(markers,colors, [0,1]):
    axbig.scatter(
        S_t_sne[labels == i, 0], S_t_sne[labels == i, 1],color=color, marker=marker,alpha=0.5, lw=2, label=i
    )
plt.legend(['label = 0','label = 1'],loc="best", shadow=False,scatterpoints=1,markerscale=1.5)
#plt.title("t-sne")

for marker,color, i in zip(markers,colors, [0]):
    axs[0,1].scatter(
        S_t_sne[labels == i, 0], S_t_sne[labels == i, 1],color="navy", marker='x',alpha=0.5, lw=2, label=i
    )
for marker,color, i in zip(markers,colors, [1]):
    axs[1,1].scatter(
        S_t_sne[labels == i, 0], S_t_sne[labels == i, 1],color="darkorange", marker='+',alpha=0.5, lw=2, label=i
    )
axbig.legend(['label = 0','label = 1'],loc="best", shadow=False,scatterpoints=1,markerscale=1.5)
axs[0,1].set_title('label = 0')
axs[1,1].set_title('label = 1')

axbig.set_xticks([])
axbig.set_yticks([])
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])
fig.tight_layout()


# %%
