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
sys.path.append('..')
from model_input import model_input
df = pd.read_feather('../output/train_test2')
sys.path.append('../output')
sys.path.append('../test')
cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df)

from gb_q_hyp import param, idx


# %%
state = pd.read_feather('../output/stateclean')
state_results = pd.read_feather('../output/state_results')
state_results=state_results.divide(state.max_mrms.values,axis=0)
state['qgb_t 0.50'] = state_results['qgb_t 0.50'].values
#%%
# Calculate the quantiles
top_quantile = state['qgb_t 0.50'].quantile(0.9)
bottom_quantile = state['qgb_t 0.50'].quantile(0.1)
print(top_quantile)
print(bottom_quantile)

# Filter the DataFrame for the top and bottom 10%
state_bad = state[state['qgb_t 0.50'] >= top_quantile]
state_good = state[state['qgb_t 0.50'] <= bottom_quantile]

state_bad = state_bad.drop(columns='qgb_t 0.50')
state_good = state_good.drop(columns='qgb_t 0.50')
perm = list(all_permutations[idx])
state = state.iloc[:,perm]

#%%
fig, axs = plt.subplots(2,5, figsize=(14,5), facecolor='w', edgecolor='k',sharex=True)
fig.subplots_adjust(hspace = .18, wspace=.4)

axs = axs.ravel()

make_log = [0,1]

state = state[['std_int_point','median_int_point','mrms_lat', 
       'duration', 'point_elev', 'point_aspect','rqi_min',  'rqi_std',  'area',
       'velocity']]

title =['intensity std dev','intensity median','latitude', 
       'duration', 'elevation', 'aspect','RQI min',  'RQI std dev',  'area',
       'velocity']
for i,col in enumerate(state.columns):


    d = [state_good[col],state_bad[col]]

    axs[i].boxplot(d)
    
    axs[i].set_xticks([1,2],labels=['low',
                                        'high'],rotation=45)
    axs[i].set_title(title[i])

    test1 = np.isin(i,make_log)
    
    if test1 == True:
        axs[i].set_yscale('log')

#%% 29 columns
    
fig.savefig("../output_figures/spread.pdf",
       bbox_inches='tight',dpi=255,transparent=False,facecolor='white')