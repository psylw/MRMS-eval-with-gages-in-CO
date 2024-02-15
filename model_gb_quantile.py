
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
from model_input import model_input

df = pd.read_feather('output/train_test2')

cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df)

#%%
# start with default hyperparameters 
# quantile loss
all_models = {}

for alpha in [0.05, 0.5, 0.95]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha)
    all_models["qgb %1.2f" % alpha] = gbr
'''for alpha in [0.05, 0.5, 0.95]:
    q = QuantileRegressor(quantile=alpha, solver='highs')
    all_models["qlin %1.2f" % alpha] = q'''



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
'''for name, gbr in sorted(all_models.items()):
    print(name)
    for idx,perm in enumerate(all_permutations):
        print(idx/len(all_permutations))
        x = cross_validate(gbr,X_train[:,perm],y_train, cv = cv,
                        scoring=scoring)
        print(x['test_neg_05p'].mean())
        print(x['test_neg_5p'].mean())
        print(x['test_neg_95p'].mean())
        scores.append([name,idx,
                    x['test_neg_05p'].mean(),x['test_neg_05p'].std(),
                    x['test_neg_5p'].mean(),x['test_neg_5p'].std(),
                    x['test_neg_95p'].mean(),x['test_neg_95p'].std(),

                    ])
scores = pd.DataFrame(scores,columns=['names','IDX',
                              'test_neg_05p_mean','test_neg_05p_std',
                              'test_neg_5p_mean','test_neg_5p_std',
                              'test_neg_95p_mean','test_neg_95p_std'])'''

#scores.to_feather('output/scores_allperm')
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
#scores.to_feather('output/scores_allperm')
# %%
# WHAT FEATURES ARE BEST FOR EACH QUANTILE?? LOOK AT FEATURES
# select optimal noncorrelated permutation per model
'''model_perm = []

for name, score in zip(all_models.keys(),['test_neg_05p_mean','test_neg_5p_mean','test_neg_95p_mean']):
    model_perm.append(scores.loc[scores.names==name].sort_values(score).iloc[-1])

model_perm = pd.concat(model_perm,axis=1)'''

#%%
print(scores.iloc[scores.test_score_mean.abs().argmin()])
idx = int(scores.iloc[scores.test_score_mean.abs().argmin()].IDX)
print(idx)
idx = 362

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
