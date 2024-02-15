#%%

import pandas as pd

#import pandas as pd
df = pd.read_feather('../output/window_values_new')
#%%
df = df.loc[(df.mrms_lat!=40.57499999999929)&(df.mrms_lon!=254.91499899999639)]
print(len(df))
df = df.dropna()
print(len(df))
#%%
# open window values
accumulation_threshold = 1
df = df.loc[(df.total_mrms_accum>accumulation_threshold)|(df.total_gage_accum>accumulation_threshold)]

df['onoff'] = 0
df.loc[(df.total_mrms_accum>0)&(df.total_gage_accum>0),['onoff']]='TP'
df.loc[(df.total_mrms_accum==0)&(df.total_gage_accum>0),['onoff']]='FN'
df.loc[(df.total_mrms_accum>0)&(df.total_gage_accum==0),['onoff']]='FP'

df.groupby('onoff').count()/len(df)*100
# %%

len(df.loc[df.total_mrms_accum>df.total_gage_accum])/len(df)

# %%
import numpy as np
df['max_gage'] = [np.max(df['15_int'][i]) for i in df.index]
df['max_mrms'] = [np.max(df['mrms_15_int'][i]) for i in df.index]