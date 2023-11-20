# %%
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

sys.path.append('../class')

from NRMSD import nrmsd

# open both window values
file_path = os.path.join('..', 'output', 'window_values_FN')
ms = pd.read_feather(file_path)

file_path = os.path.join('..', 'output', 'window_values_radaronly')
radar = pd.read_feather(file_path)

accumulation_threshold = 0

ms = ms.loc[(ms.total_accum_atgage>accumulation_threshold)|(ms.total_gage_accum>accumulation_threshold)]

radar = radar.loc[(radar.total_accum_atgage>accumulation_threshold)|(radar.total_gage_accum>accumulation_threshold)]
# %%
# calculate mce
ms_nrmsd = nrmsd(ms)
radar_nrmsd = nrmsd(radar)
# %%
######################   radar VALUES    ###############################################################################################

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
for i in range(100):
    i = random.randint(0, len(radar))
    
    c = radar.iloc[i]
    
    axs[0].scatter(c.gage,c.mrms)
    axs[0].set_title('radar only')
    
    t = ms.loc[(ms['index']==c['index'])&(ms['gage_id']==c['gage_id'][0])]
    
    try:
        axs[1].scatter(t.gage.iloc[0],t.mrms.iloc[0])
        axs[1].set_title('with correction')
    except:
        pass
# %%
######################   COMPARE TIME SERIES    ###############################################################################################
for i in range(100):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    c = radar.iloc[i]
    
    axs[0].plot(c.mrms,label='mrms')
    axs[0].plot(c.gage,label='gage')
    axs[0].legend()
    axs[0].set_title(str(c.mce)[0:4])
    
    t = ms.loc[(ms['index']==c['index'])&(ms['gage_id']==c['gage_id'][0])]
    try:
        axs[1].plot(t.mrms.iloc[0],label='mrms')
        axs[1].plot(t.gage.iloc[0],label='gage')
        axs[1].legend()
        axs[1].set_title(str(t.mce.iloc[0])[0:4])
    except:
        pass
    plt.show()

# %%

ms['nrmsd']=ms_nrmsd
radar['nrmsd']=radar_nrmsd

# %%
''''''
accumulation_threshold = 25

ms_thresh = ms.loc[(ms.total_accum_atgage>accumulation_threshold)|(ms.total_gage_accum>accumulation_threshold)]

radar_thresh = radar.loc[(radar.total_accum_atgage>accumulation_threshold)|(radar.total_gage_accum>accumulation_threshold)]
# %%

print(len(ms_thresh))
print(len(radar_thresh))

end = 1
step = .1

h_ms = np.histogram(ms_thresh.nrmsd,bins=np.arange(0,end, step))
h_r = np.histogram(radar_thresh.nrmsd,bins=np.arange(0,end, step))


import matplotlib as mpl

# set figure defaults
mpl.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12.0/2, 10.0/2)

fig, ax = plt.subplots(figsize=(6, 6))


plt.xlabel('NRMSD')
plt.ylabel('frequency')

plt.bar(h_ms[1][:-1],h_ms[0].astype(float)/len(ms_thresh),edgecolor = 'blue', color = [], width = step-.02, linewidth = 2,label='with correction')
plt.bar(h_r[1][:-1],h_r[0].astype(float)/len(radar_thresh),edgecolor = 'r', color = [], width = step-.02
, linewidth = 2,label='radar only')

plt.legend()

# %%
