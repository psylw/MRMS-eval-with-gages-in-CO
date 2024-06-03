# %%
###############################################################################
# Compare gradient boosting and linear quantile regressor cv and test results
###############################################################################
import random
import importlib
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

sys.path.append('../utils')
from model_input import model_input

# import training data
df = pd.read_feather('../output/train_test2')
state_input = pd.read_feather('../output/experiments/stateclean_year')
output_dir = '../output/experiments'

# import experiments
import sys
sys.path.append('../output/experiments')
sys.path.append('..')

from experiments2 import *

# select which experiments to run
experiments = [
    #original,
    #pre_2021,
    #post_2021,
    #jja,
    #pre_2021_jja,
    #post_2021_jja,
    #add_year_as_feature,
    nr_rmse,
    nr_rmse_pre,
    nr_rmse_post,
    mean_error,
    mean_error_pre,
    mean_error_post,
    mean_error_pre_2021_jja,
    mean_error_post_2021_jja,
    mean_error_year
]

# define loss
scoring={'neg_05p': make_scorer(
    mean_pinball_loss,
    alpha=.05,
),
'neg_5p': make_scorer(
    mean_pinball_loss,
    alpha=.5,
),
'neg_95p': make_scorer(
    mean_pinball_loss,
    alpha=.95,
)}

#%%
for idx, exp in enumerate(experiments):
    # define train/test data
    df_experiment, state = exp(df,state_input)

    cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df_experiment)
    #########################

    # get hyperparameters
    filename = f"{exp.__name__ }"
    param = importlib.import_module(filename).param
    idx = importlib.import_module(filename).idx
    ######################
    
    # define model
    all_models = {}
    for alpha, p in zip([0.05, 0.5, 0.95],param):
        gbr_t = GradientBoostingRegressor(**p,loss="quantile", alpha=alpha)
        all_models["qgb_t %1.2f" % alpha] = gbr_t

    for alpha in [0.05, 0.5, 0.95]:
        q = QuantileRegressor(quantile=alpha, solver='highs',alpha=0.02)
        all_models["qlin %1.2f" % alpha] = q
    #############################################################
    # plot correlation matrix of highest performing permutation

    pltcorr = train.drop(columns=['norm_diff']).iloc[:,list(all_permutations[idx])]
    pltcorr['norm_diff'] = train.norm_diff

    plt.figure(figsize=(8, 6))

    correlation_matrix = pltcorr.corr(method='spearman')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()
    ###########################################################
    # get training scores
    scores=[]
    i=0
    for name, gbr in zip(all_models.keys(),list(all_models.values())):
        x = cross_validate(gbr,X_train[:,all_permutations[idx]],y_train, cv = cv,
                        scoring=scoring)

        if i == 0 or i == 3:
            scores.append([name, x['test_neg_05p'].mean(),x['test_neg_05p'].std()])
        elif i == 1 or i == 4:
            scores.append([name, x['test_neg_5p'].mean(),x['test_neg_5p'].std()])
        else:
            scores.append([name, x['test_neg_95p'].mean(),x['test_neg_95p'].std()])
        i+=1

    scores = pd.DataFrame(scores,columns=['name','mean_ql','std_ql'])
    ##########################################################################
    # get fraction of outliers
    train_results=[]
    for i in range(5):
        d={}
        for name, gbr,a in zip(all_models.keys(),list(all_models.values()),[0.05,.5,.95,0.05,.5,.95]):
            c = cv[i]
            gbr.fit(X_train[:,all_permutations[idx]][cv[i][0]],y_train[cv[i][0]])

            pred = gbr.predict(X_train[:,all_permutations[idx]][cv[i][1]])

            d[name] = pred
            
        d = pd.DataFrame(d)
        d['truth'] = y_train[cv[i][1]] 
        d['cv'] = i
        train_results.append(d)

    cv_ci_gbt=[]
    cv_ci_l=[]
    for i in range(5):
        c = train_results[i]

        cv_ci_gbt.append(len(c.loc[(c.truth>c['qgb_t 0.95'])|(c.truth<c['qgb_t 0.05'])])/len(c))
        cv_ci_l.append(len(c.loc[(c.truth>c['qlin 0.95'])|(c.truth<c['qlin 0.05'])])/len(c))

    new_row = {"name": "gbt", "mean_ql": np.mean(cv_ci_gbt), "std_ql": np.std(cv_ci_gbt)}
    scores = scores.append(new_row, ignore_index=True)

    new_row = {"name": "lin", "mean_ql": np.mean(cv_ci_l), "std_ql": np.std(cv_ci_l)}
    scores = scores.append(new_row, ignore_index=True)

    scores.round(3).to_feather('../output/experiments/train_'+filename)
    #######################################################################
    # get test results and predictions
    test_results={}
    test_perf = []
    for name, gbr,a in zip(all_models.keys(),list(all_models.values()),[.05,.5,.95,.05,.5,.95]):
        gbr.fit(X_train[:,all_permutations[idx]],y_train)
        pred = gbr.predict(X_test[:,all_permutations[idx]])
        test_results[name]=pred
        test_perf.append([name,mean_pinball_loss(y_test,pred,alpha=a)])

    test = pd.DataFrame(test_perf,columns=['name','ql'])

    test_results = pd.DataFrame(test_results)
    test_results['truth'] = y_test

    new_row = {"name": "gbt", "ql": len(test_results.loc[(test_results.truth>test_results['qgb_t 0.95'])|(test_results.truth<test_results['qgb_t 0.05'])])/len(test_results)}
    test = test.append(new_row, ignore_index=True)

    new_row = {"name": "lin", "ql": len(test_results.loc[(test_results.truth>test_results['qlin 0.95'])|(test_results.truth<test_results['qlin 0.05'])])/len(test_results)}
    test = test.append(new_row, ignore_index=True)

    test.round(3).to_feather('../output/experiments/test_'+filename)

    # save training predictions
    test_results.to_feather('../output/experiments/testpredictions_'+filename)

    ##################################################
    # get state results

    scaler = StandardScaler()
    X_state = scaler.fit_transform(state)

    state_results={}
    for name, gbr in zip(all_models.keys(),list(all_models.values())):
        gbr.fit(X_train[:,all_permutations[idx]],y_train)
        pred = gbr.predict(X_state[:,all_permutations[idx]])
        state_results[name]=pred
    state_results = pd.DataFrame(state_results)
    state_results.to_feather('../output/experiments/state_results_'+filename)
# %%
