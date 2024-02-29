# %%
# add results and save
# run hist_rmse.py first to get results
import pandas as pd

df_state = pd.read_feather('../output/state').dropna()
df_train = pd.read_feather('../output/train_test2').dropna()
df_window = pd.read_feather('../output/window_values_new').dropna()

df_state = df_state.loc[df_state.rqi_min>=0]

#%%

#test = df.loc[df.norm_diff>70]
#test.groupby(['start','mrms_lat','mrms_lon']).count()
# this gage has most big outliers, see code above, remove it
df_state = df_state.loc[(df_state.mrms_lat!=40.57499999999929)&(df_state.mrms_lon!=254.91499899999639)]
df_train= df_train.loc[(df_train.mrms_lat!=40.57499999999929)&(df_train.mrms_lon!=254.91499899999639)]
df_window = df_window.loc[(df_window.mrms_lat!=40.57499999999929)&(df_window.mrms_lon!=254.91499899999639)]

df_train = df_train.loc[(df_train.total_mrms_accum>1)].reset_index(drop=True)
df_window = df_window.loc[(df_window.total_mrms_accum>1)].reset_index(drop=True)

df_window = df_window.drop(columns=['index','storm_id', 'mrms_lat', 'mrms_lon',
       'total_gage_accum', 'total_mrms_accum', 'rqi_mean', 'rqi_median',
       'rqi_min', 'rqi_max', 'rqi_std', 'norm_diff'])

# %%
# add results and save
# run hist_rmse.py first to get results
df_state_results = pd.read_feather('../output/state_results')

df_state_results = pd.concat([df_state.reset_index(drop=True),df_state_results],axis=1).drop(columns='index')
df_state_results.to_feather('../output/final_state_output')


df_train_window = pd.concat([df_window,df_train],axis=1)
df_train_window.to_feather('../output/final_train_output')