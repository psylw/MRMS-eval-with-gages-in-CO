###############################################################################
# train gradient boosting regressor with quantile loss
# determine best performing subsection of data that has low feature correlation for alpha = 0.5
# tune hyperparameters for each alpha value
###############################################################################

# used this tutorial to explore quantile loss for gb

#https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html


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

sys.path.append('utils')
from model_input import model_input

# import training data
df = pd.read_feather('output/train_test2')
state_input = pd.read_feather('output/experiments/stateclean_year')
output_dir = 'output/experiments'

# import experiments
from experiments import *

# select which experiments to run
experiments = [
    original,
    pre_2021,
    post_2021,
    jja,
    pre_2021_jja,
    post_2021_jja,
    add_year_as_feature,
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
#%%
# define model and loss
# start with default hyperparameters 
# quantile loss
all_models = {}

for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha)
    all_models["qgb %1.2f" % alpha] = gbr

scoring={'neg_05p': make_scorer(
    mean_pinball_loss,
    alpha=.05,
    greater_is_better=False,  # maximize the negative loss
),
'neg_5p': make_scorer(
    mean_pinball_loss,
    alpha=.5,
    greater_is_better=False,  # maximize the negative loss
),
'neg_95p': make_scorer(
    mean_pinball_loss,
    alpha=.95,
    greater_is_better=False,  # maximize the negative loss
)}

# define parameter space to explore
param_space = {
'n_estimators': randint(50, 250),  # Number of boosting stages to be run
'learning_rate': uniform(0.01, .9),  # Learning rate
'max_depth': randint(2, 5),  # Maximum depth of the individual trees
'min_samples_split': randint(2, 20),  # Minimum samples required to split a node
'min_samples_leaf': randint(1, 20),  # Minimum samples required at each leaf node
#'subsample':  [0.5, 0.7, 0.9, 1.0],
    'random_state': randint(0,1000)  # Number of features to consider at each split
}
#%%
###########################################################
# run through each experiment, save best set of features and hyperparameters

for idx, exp in enumerate(experiments):
    print(idx/len(experiments))
    df_experiment,_ = exp(df,state_input)

    # split into cross folds and test set, apply standard scalar, select features
    cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df_experiment)

    # find best feature set for alpha = 0.5
    gbr = list(all_models.items())[1][1]

    scores = []
    for perm in all_permutations:
        x = cross_validate(gbr,X_train[:,perm],y_train, cv = cv,
                        scoring=scoring['neg_5p'])

        scores.append(x['test_score'].mean())

    idx = int(np.argmin(np.abs(scores)))

    # TUNE HYPERPERAMETERS FOR EACH QUANTILE
    hyp = []
    for gbr, score in zip(list(all_models.values()),list(scoring)):
        mod = RandomizedSearchCV(estimator=gbr,
                        param_distributions = param_space,
                        n_iter=60, 
                        scoring=scoring[score],
                        n_jobs=10,
                        random_state=42,
                        cv=cv)

        _ = mod.fit(X_train[:,all_permutations[idx]],y_train)  
        
        hyp.append(pd.DataFrame(mod.cv_results_))
    # add hyperparams
    param = []
    for i in range(len(hyp)):
        param.append(hyp[i].sort_values(by='rank_test_score').iloc[0].params)

    filename = f"{exp.__name__ }.py"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, 'w') as file:
        file.write(f"param = {repr(param)}\n")
        file.write(f"idx = {repr(idx)}\n")


# %%
