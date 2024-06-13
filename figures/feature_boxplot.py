# open training data and statewide results
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

def feature_boxplot(idx,state,state_results,train, all_permutations,plot_name,fig_idx):

       state = state.copy()
       state['qgb_t 0.50'] = state_results['qgb_t 0.50'].values
       readable_names = {
       'mrms_lat': 'latitude', 
       'mrms_lon': 'longitude', 
       'total_mrms_accum': 'total accum', 
       'rqi_mean': 'RQI mean', 
       'rqi_median': 'RQI median',
       'rqi_min': 'RQI min',
       'rqi_max': 'RQI max', 
       'rqi_std': 'RQI std dev', 
       'max_mrms': 'max intensity',
       'max_accum_atgage': 'max accum',
       'median_int_point': 'intensity median',
       'std_int_point': 'intensity std dev',
       'var_int_point': 'var intensity',
       'mean_int_point': 'mean intensity',
       'median_accum_point': 'median accum',
       'std_accum_point': 'std dev accum',
       'var_accum_point': 'var accum',
       'mean_accum_point': 'mean accum',
       'point_elev': 'elevation',
       'point_slope': 'slope',
       'point_aspect': 'aspect',
       'temp_var_accum': 'storm temp var',
       'spatial_var_accum': 'storm spatial var',
       }
       state = state.rename(columns=readable_names)
       #%%
       # Calculate the quantiles
       top_quantile = state['qgb_t 0.50'].quantile(0.9)
       bottom_quantile = state['qgb_t 0.50'].quantile(0.1)
       print(top_quantile)
       print(bottom_quantile)

       # Filter the DataFrame for the top and bottom 10%
       state_bad = state.loc[state['qgb_t 0.50'] >= top_quantile]
       state_good = state.loc[state['qgb_t 0.50'] <= bottom_quantile]
       print(len(state_bad))
       print(len(state_good))
       #%%
       state_bad = state_bad.drop(columns='qgb_t 0.50')
       state_good = state_good.drop(columns='qgb_t 0.50')


       #%%
       fig, axs = plt.subplots(2,5, figsize=(14*.75,5*.85), facecolor='w', edgecolor='k',sharex=True)
       fig.subplots_adjust(hspace = .18, wspace=.4)

       axs = axs.ravel()

       make_log = {
       'latitude' : False, 
       'longitude' : False, 
       'total accum' : True, 
       'RQI mean' : False, 
       'RQI median' : False,
       'RQI min' : False,
       'RQI max' : False, 
       'RQI std dev' : False, 
       'max intensity' : True,
       'max accum' : True,
       'intensity median' : True,
       'intensity std dev' : False,
       'var intensity' : False,
       'mean intensity' : True,
       'median accum' : True,
       'std dev accum' : False,
       'var accum' : False,
       'mean accum' : True,
       'elevation' : False,
       'slope' : False,
       'aspect' : False,
       'storm temp var' : False,
       'storm spatial var' : False,
       'duration' : False,
       'month' : False,
       'hour' : False,
       'velocity' : False,
       'area' : True
       }

       columns = train.drop(columns='norm_diff').iloc[:,list(all_permutations[idx])].rename(columns=readable_names)

       columns = columns.drop(columns=['month','hour','year'], errors='ignore').columns
       flierprops = dict(marker='o', markerfacecolor='r', markersize=4, linestyle='none')
       for i,col in enumerate(columns):
              d = [state_good[col],state_bad[col]]

              axs[i].boxplot(d, flierprops=flierprops, patch_artist=True, 
              boxprops=dict(facecolor='lightblue', color='blue'),
              medianprops=dict(color='red', linewidth=2),
              whiskerprops=dict(color='blue', linewidth=1.5),
              capprops=dict(color='blue', linewidth=1.5))
              
              axs[i].set_xticks([1,2],labels=['low',
                                                 'high'],rotation=45,fontsize=12)
              #axs[i].set_yticks(fontsize=12)
              axs[i].set_title(col,fontsize=12)
              axs[i].grid(True, linestyle='--', linewidth=0.5, axis='y')
              if make_log[col] == True:
                     axs[i].set_yscale('log')

       #%% 29 columns
       #fig.suptitle(plot_name, fontsize=16)
       #fig.savefig("../output_figures/f05.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
       plt.tight_layout()
       fig.savefig("../output_figures/experiments/S"+str(fig_idx)+".pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
