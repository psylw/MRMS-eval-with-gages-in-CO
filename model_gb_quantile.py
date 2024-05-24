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
############################################################
# EXPERIMENTS

# 1. pre 2021 ############################################################
#df['year'] = [df.start[i].year for i in df.index]
#df = df[df.year<2021]
#df = df.drop(columns='year')

# 2. post 2021 ############################################################
#df['year'] = [df.start[i].year for i in df.index]
#df = df[df.year>=2021]
#df = df.drop(columns='year')

# 3. JJA ############################################################
#df = df[df.month.isin([6,7,8])]

# 4. pre 2021 JJA ############################################################
#df = df[df.month.isin([6,7,8])]
#df['year'] = [df.start[i].year for i in df.index]
#df = df[df.year<2021]
#df = df.drop(columns='year')

# 5. post 2021 JJA ############################################################
#df = df[df.month.isin([6,7,8])]
#df['year'] = [df.start[i].year for i in df.index]
#df = df[df.year>=2021]
#df = df.drop(columns='year')

# 6. add year as feature ############################################################
# df['year'] = [df.start[i].year for i in df.index]

# 7. nRMSE ############################################################
#df['norm_diff'] = df['norm_diff']/df['max_mrms']

# 8. mean error ############################################################
#df = df.loc[df.total_mrms_accum>1].reset_index(drop=True)
#df = df.dropna()
#df['norm_diff'] = pd.read_feather('output/mean_error')

# 9. mean error pre 2021 JJA ############################################################
#df = df[df.month.isin([6,7,8])]
#df['year'] = [df.start[i].year for i in df.index]
#df = df[df.year<2021]
#df = df.drop(columns='year')
#df = df.loc[df.total_mrms_accum>1].reset_index(drop=True)
#df = df.dropna()
#df['norm_diff'] = pd.read_feather('output/mean_error')

# 10. mean error post 2021 JJA ############################################################
#df = df[df.month.isin([6,7,8])]
#df['year'] = [df.start[i].year for i in df.index]
#df = df[df.year>=2021]
#df = df.drop(columns='year')
#df = df.loc[df.total_mrms_accum>1].reset_index(drop=True)
#df = df.dropna()
#df['norm_diff'] = pd.read_feather('output/mean_error')

############################################################

#%%
###########################################################
# split into cross folds and test set, apply standard scalar, select features
cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df)

#%%
# start with default hyperparameters 
# quantile loss
all_models = {}

for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha)
    all_models["qgb %1.2f" % alpha] = gbr

# %%
# WHAT FEATURES ARE BEST FOR EACH QUANTILE??
#SAMPLE_permutations = random.sample(all_permutations, 10) # change to all perms
scores = []

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

# %%

gbr = list(all_models.items())[1][1]
print(gbr)
for idx,perm in enumerate(all_permutations):
    print(idx/len(all_permutations))
    x = cross_validate(gbr,X_train[:,perm],y_train, cv = cv,
                    scoring=scoring['neg_5p'])

    print(x['test_score'].mean())

    scores.append([idx,
                x['test_score'].mean(),x['test_score'].std(),
                ])
scores = pd.DataFrame(scores,columns=['IDX',
                            'test_score_mean','test_score_std'])

#%%
print(scores.iloc[scores.test_score_mean.abs().argmin()])
idx = int(scores.iloc[scores.test_score_mean.abs().argmin()].IDX)
print(idx)

#%%
# TUNE HYPERPERAMETERS FOR EACH QUANTILE
    #gbc
param = {
'n_estimators': randint(50, 250),  # Number of boosting stages to be run
'learning_rate': uniform(0.01, .9),  # Learning rate
'max_depth': randint(2, 5),  # Maximum depth of the individual trees
'min_samples_split': randint(2, 20),  # Minimum samples required to split a node
'min_samples_leaf': randint(1, 20),  # Minimum samples required at each leaf node
#'subsample':  [0.5, 0.7, 0.9, 1.0],
    'random_state': randint(0,1000)  # Number of features to consider at each split
}

hyp = []
for gbr, score in zip(list(all_models.values()),list(scoring)):
    print(score)
    mod = RandomizedSearchCV(estimator=gbr,
                       param_distributions = param,
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

print(param)
#%%
