
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('class')
from mce import *

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

mce_sorted = mce(df)



# %%
test = df.drop(columns = ['gage_id', 'mce','mrms_accum_atgage','gage_accum', 
       'onoff', 'mrms', 'gage', 
       'total_gage_accum'])

high = test.loc[(test.norm_diff>.6)&(test.total_accum_atgage>5)]
#low = df.loc[(df.mce<.5)&(df.mce>0)&(df.total_accum_atgage>10)]
low = test.loc[(test.norm_diff<0)&(test.total_accum_atgage>5)].sample(n=len(high))
# %%
for i in high.columns:
    d = [high[i],low[i]]
    plt.boxplot(d)
    plt.xticks([1,2],labels=['mce>0.6','mce<0'],rotation=45)
    plt.title(i)
    plt.show()
