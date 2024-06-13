#%%
import pandas as pd
import sys
import importlib
import numpy as np
sys.path.append('../utils')
sys.path.append('../test')
sys.path.append('../output/experiments')
sys.path.append('..')

from model_input import model_input

# import training data
df = pd.read_feather('../output/train_test2')
state_input = pd.read_feather('../output/experiments/stateclean_year')
output_dir = '../output/experiments'

# import experiments
from experiments2 import *

# select which experiments to run
experiments = [
    #original,
    pre_2021,
    #post_2021,
    #jja,
    #pre_2021_jja,
    #post_2021_jja,
    #add_year_as_feature,
    #nr_rmse,
    #nr_rmse_pre,
    #nr_rmse_post,
    #mean_error,
    #mean_error_pre,
    #mean_error_post,
    #mean_error_pre_2021_jja,
    #mean_error_post_2021_jja,
    #mean_error_year
]

# import figure functions
sys.path.append('figures')
from hist_rmse import hist_rmse
from map_results import map_results
from perm_imp import perm_imp
from feature_boxplot import feature_boxplot
from predictedvstruth import predictedvstruth


#%%
fig_idx = 7
for idx, exp in enumerate(experiments):
    print(idx/len(experiments))
    file_name = exp.__name__ 

    # open variables required for plots
    df_experiment, state = exp(df,state_input)

    # split into cross folds and test set, apply standard scalar, select features
    cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df_experiment)

    test_results = pd.read_feather('../output/experiments/testpredictions_'+file_name)
    
    state_results = pd.read_feather('../output/experiments/state_results_'+file_name)

    hist_rmse(test_results,state_results,train,test,state, file_name,fig_idx)
    #fig_idx+=1
    #map_results(state, state_results,file_name)

    param = importlib.import_module(file_name).param
    idx = importlib.import_module(file_name).idx
    #perm_imp(idx,param,train,X_train, X_test, y_train, y_test, all_permutations,file_name,fig_idx)
    #fig_idx+=1
    #feature_boxplot(idx,state,state_results,train, all_permutations,file_name,fig_idx)
    #fig_idx+=1
    #predictedvstruth(test_results,file_name,fig_idx)

    fig_idx+=5




# %%
