# %%
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

radar = pd.read_feather('../output/window_values_new')
ms = pd.read_feather('../output/window_values_ms')
radar = radar.loc[radar.total_mrms_accum>1].reset_index(drop=True)
radar = radar.dropna()
ms = ms.loc[ms.total_mrms_accum>1].reset_index(drop=True)
ms = ms.dropna()
#test = df.loc[df.norm_diff>70]
#test.groupby(['start','mrms_lat','mrms_lon']).count()
# this gage has most big outliers, see code above, remove it
radar = radar.loc[(radar.mrms_lat!=40.57499999999929)&(radar.mrms_lon!=254.91499899999639)]

ms = ms.loc[(ms.mrms_lat!=40.57499999999929)&(ms.mrms_lon!=254.91499899999639)]


#%%
ms['max_mrms'] = [np.max(ms.mrms_15_int[i]) for i in ms.index]
ms['nrmsd'] = ms.norm_diff/ms.max_mrms
print(ms.nrmsd.median())

radar = radar.loc[radar['index'].isin(ms['index'])]
radar['max_mrms'] = [np.max(radar.mrms_15_int[i]) for i in radar.index]
radar['nrmsd'] = radar.norm_diff/radar.max_mrms
print(radar.nrmsd.median())

end = 2
step = .1

h_ms = np.histogram(ms.nrmsd,bins=np.arange(0,end, step))
h_r = np.histogram(radar.nrmsd,bins=np.arange(0,end, step))


import matplotlib as mpl

# set figure defaults
mpl.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12.0/2, 10.0/2)

fig, ax = plt.subplots(figsize=(6, 6))


plt.xlabel('RMSE')
plt.ylabel('frequency')

plt.bar(h_ms[1][:-1],h_ms[0].astype(float)/len(ms),edgecolor = 'blue', color = [], width = step-.02, linewidth = 2,label='with correction')
plt.bar(h_r[1][:-1],h_r[0].astype(float)/len(radar),edgecolor = 'r', color = [], width = step-.02
, linewidth = 2,label='radar only')

plt.legend()
fig.savefig("../output_figures/S1.pdf",
       bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
