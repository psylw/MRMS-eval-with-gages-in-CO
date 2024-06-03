#%%
# plot results for each year for different approaches to separate changes in MRMS

import pandas as pd
import sys
import importlib
import numpy as np
sys.path.append('../utils')
sys.path.append('../test')
sys.path.append('../output/experiments')
sys.path.append('..')



# import training data
df = pd.read_feather('../output/train_test2')
state_input = pd.read_feather('../output/experiments/stateclean_year')
output_dir = '../output/experiments'

# import experiments
from experiments import *

# select which experiments to run
experiments = [
    pre_2021,
    post_2021,
    pre_2021_jja,
    post_2021_jja,
    add_year_as_feature,
    mean_error_pre_2021_jja,
    mean_error_post_2021_jja,
    mean_error_year
]

# import figure functions
sys.path.append('../figures')
from map_results import map_results

#%%

for idx, exp in enumerate(experiments):
    file_name = exp.__name__ 
    print(file_name)

    # open variables required for plots
    df_experiment, state = exp(df,state_input)
    state = state.reset_index()
    state_results = pd.read_feather('../output/experiments/state_results_'+file_name)

    input_state_results=state_results.divide(state.max_mrms.values,axis=0)

    state = pd.concat([state,input_state_results],axis=1)
    print(len(state))
    
    if 'mean_error' in file_name:
        levels = list(np.arange(-.25,.5,.1))
    else:
        levels = list(np.arange(0.3,0.8,.1))

    for year in state.year.unique():
        print(year)
        plot = state[state.year==year]

        map_results(plot, levels,file_name)

# %%
