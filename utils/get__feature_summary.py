# %%
import pandas as pd
import os

codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)

df = pd.read_feather(parentDir+'train_test')
df_state = pd.read_feather(parentDir+'predict')

order = ['max_mrms', 'median_int_point', 'std_int_point','var_int_point', 'mean_int_point',
         'total_accum_atgage','median_accum_point','std_accum_point', 'var_accum_point', 'mean_accum_point',
        'duration','month', 'hour',
       'latitude', 'longitude',  
         'mult_correct', 'RQI',
       'point_elev', 'point_aspect','point_slope', 
         'max', 'median','std', 'var', 'mean', 
         'accum_max_storm','accum_median_storm', 'accum_std_storm', 'accum_var_storm','accum_mean_storm', 
         'duration_storm', 
         'area', 'axis_major_length', 'axis_minor_length',
         'eccentricity', 'velocity',
       'elevation_foot', 'storm_elev',  'storm_aspect','slope_foot'
       ]

get_summary=df.reindex(columns=order)
get_summary_state=df_state.reindex(columns=order)

# %%
for i in get_summary.columns:
    print(get_summary[i].mean())