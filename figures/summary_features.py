# %%

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, randint
import seaborn as sns
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,RandomizedSearchCV

from sklearn.metrics import  mean_absolute_error,r2_score,mean_pinball_loss, mean_squared_error,mean_pinball_loss


df = pd.read_feather('../output/train_test2')
df['year'] = [df.start[i].year for i in df.index]
df_state = pd.read_feather('../output/state')
df_state = df_state.loc[df_state.rqi_min>=0]
df_state['year'] = [df_state.time[i].year for i in df_state.index]
#%%
df = df.dropna()
df_state = df_state.dropna()
#test = df.loc[df.norm_diff>70]
#test.groupby(['start','mrms_lat','mrms_lon']).count()
# this gage has most big outliers, see code above, remove it
df = df.loc[(df.mrms_lat!=40.57499999999929)&(df.mrms_lon!=254.91499899999639)]
df_state = df_state.loc[(df_state.mrms_lat!=40.57499999999929)&(df_state.mrms_lon!=254.91499899999639)]

df = df.loc[(df.total_mrms_accum>1)].reset_index(drop=True)

df = df.drop(columns=['start','storm_id'])


#%%
df_state=df_state.reindex(columns=df.drop(columns=['norm_diff']).columns)
df_state = df_state.reset_index(drop=True)
#df_state.to_feather('../output/stateclean')
df_state.to_feather('../output/stateclean_year')
# %%
for f in df_state.columns:
    #df_state[f].hist()
    #plt.title(f)
    #plt.show()
    #df[f].hist()
    #plt.title(f)
    #plt.show()
    print(df_state[f].mean())
    print(df[f].mean())
# %%
df = df.reindex(columns=['max_mrms',
                    'median_int_point', 
                    'std_int_point',
                    'var_int_point', 
                    'mean_int_point', 
                    'max_accum_atgage', 
                    'total_mrms_accum', 
                    'median_accum_point',
                    'std_accum_point', 
                    'var_accum_point', 
                    'mean_accum_point', 
                    'duration',
                    'month', 
                    'hour', 
                    'mrms_lat', 
                    'mrms_lon', 
                    'rqi_min', 
                    'rqi_max', 
                    'rqi_mean', 
                    'rqi_median',
                    'rqi_std',
                    'point_elev',
                    'point_aspect',
                    'point_slope', 
                    'temp_var_accum', 
                    'spatial_var_accum', 
                    'area',
                    'velocity'
                    ])
df_state=df_state.reindex(columns=df.columns)
# %%
for f in df_state.columns:
    print(df[f].mean())
# %%
for f in df_state.columns:
    print(df_state[f].mean())
# %%
for f in df_state.columns:
    print(df[f].std())
# %%
for f in df_state.columns:
    print(df_state[f].std())

# %%
