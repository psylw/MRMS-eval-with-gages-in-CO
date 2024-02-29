# %%
###############################################################################
# evaluate performance of several model types for deterministic regression
###############################################################################
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
#%%
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,RandomizedSearchCV

from sklearn.metrics import  mean_absolute_error,r2_score,mean_pinball_loss, mean_squared_error
sys.path.append('utils')
from model_input import model_input

df = pd.read_feather('output/train_test2')
sys.path.append('output')
sys.path.append('test') 
cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df)


# %%
names = [
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Bagged Tree",
    "Neural Net",
    "AdaBoost",
    "linear",
    "xgboost",
    "Gradient Boosting",
    "SVC"
]

classifiers = [
    KNeighborsRegressor(),
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(random_state=0,n_jobs=-1),
    BaggingRegressor(random_state=0,n_jobs=-1),
    MLPRegressor(random_state=0),
    AdaBoostRegressor(random_state=0),
    LinearRegression(),
    xgb.XGBRegressor(random_state=0,n_jobs=-1),
    GradientBoostingRegressor(random_state=0),
    SVR()
]
# %%

scores = []

for name, clf in zip(names, classifiers):
    print(name)
    x = cross_validate(clf,X_train,y_train, cv = cv,
                    scoring=['r2','neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'])
    print(x['test_r2'])
    print(x['test_neg_mean_absolute_error'])
    print(x['test_neg_mean_squared_error'])
    print(x['test_neg_root_mean_squared_error'])

    print(x['test_r2'].mean())
    print(x['test_neg_mean_absolute_error'].mean())
    print(x['test_neg_mean_squared_error'].mean())
    print(x['test_neg_root_mean_squared_error'].mean())

    print(x['test_r2'].std())
    print(x['test_neg_mean_absolute_error'].std())
    print(x['test_neg_mean_squared_error'].std())
    print(x['test_neg_root_mean_squared_error'].std())

    scores.append([name,
                    x['test_r2'].mean(),
                    x['test_r2'].std(),
                    x['test_neg_mean_absolute_error'].mean(),
                    x['test_neg_mean_absolute_error'].std(),
                    x['test_neg_mean_squared_error'].mean(),
                    x['test_neg_mean_squared_error'].std(),                                                                               x['test_neg_root_mean_squared_error'].mean(),
                    x['test_neg_root_mean_squared_error'].std()])

scores = pd.DataFrame(scores,columns=['names',
                            'test_r2_mean','test_r2_std',
                            'test_neg_mean_absolute_error_mean','test_neg_mean_absolute_error_std',
                            'test_neg_mean_squared_error_mean','test_neg_mean_squared_error_std',
                            'test_neg_root_mean_squared_error_mean','test_neg_root_mean_squared_error_std'])

#%%
# initial hyperparameter tuning
names = [
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Bagged Tree",
    "Neural Net",
    "AdaBoost",

    "xgboost",
    "Gradient Boosting",
    "SVC"
]

classifiers = [
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_jobs=-1),
    BaggingRegressor(n_jobs=-1),
    MLPRegressor(),
    AdaBoostRegressor(),

    xgb.XGBRegressor(n_jobs=-1),
    GradientBoostingRegressor(),
    SVR()
]

from hyperparam import param
param = param
hyp = {}
for name, clf, param in zip(names, classifiers, param):
    print(name)
    clf = clf

    mod = RandomizedSearchCV(estimator=clf,
                       param_distributions = param,
                       n_iter=15, 
                       scoring=['r2','neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'],
                       refit='neg_mean_absolute_error',
                       cv=cv)

    _ = mod.fit(X_train,y_train)  

    hyp[name] = pd.DataFrame(mod.cv_results_).sort_values('rank_test_neg_mean_absolute_error')

# %%

# add parameters to model
classifiers = [
    KNeighborsRegressor(**hyp[names[0]].params[0]),
    DecisionTreeRegressor(**hyp[names[1]].params[0]),
    RandomForestRegressor(**hyp[names[2]].params[0]),
    BaggingRegressor(**hyp[names[3]].params[0]),
    MLPRegressor(**hyp[names[4]].params[0]),
    AdaBoostRegressor(**hyp[names[5]].params[0]),
    LinearRegression(),
    xgb.XGBRegressor(**hyp[names[6]].params[0]),
    GradientBoostingRegressor(**hyp[names[7]].params[0]),
    SVR(**hyp[names[8]].params[0])
]

names = [
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Bagged Tree",
    "Neural Net",
    "AdaBoost",
    "linear",
    "xgboost",
    "Gradient Boosting",
    "SVC"
]

# %%
# what do test results look like?
#feature_names = train.drop(columns='norm_diff').columns
results = []
for name, clf in zip(names, classifiers):
    
    f, ax = plt.subplots(figsize=(6, 6))
    clf.fit(X_train,y_train)

    ypred = clf.predict(X_test)

    print(name)
    print(r2_score(y_test,ypred))
    print(mean_absolute_error(y_test,ypred))
    print(mean_squared_error(y_test,ypred))
    results.append([name,round(r2_score(y_test,ypred),2),round(mean_absolute_error(y_test,ypred),2),round(mean_squared_error(y_test,ypred),2)])
    sns.scatterplot(x=y_test, y=ypred, s=5, color=".15")
    sns.histplot(x=y_test, y=ypred, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(x=y_test, y=ypred, levels=5, color="w", linewidths=1)
    plt.xlim(0,25)
    plt.ylim(0,25)
    plt.plot([0,25],[0,25],color='red')
    plt.title(name)
    plt.show()

# %%
clf = classifiers[2]
name = names[2]
clf.fit(X_train,y_train)
ypred = clf.predict(X_test)
#%%
fig = plt.figure(figsize=(5,5))
plt.scatter(y_test, ypred,marker='+')
plt.plot([0,np.max(y_test)],[0,np.max(y_test)],color='red')
plt.title(name,fontsize=12)
plt.xlim(np.min(y_test),np.max(y_test))
plt.ylim(np.min(y_test),np.max(y_test))
plt.xlabel('true RMSE',fontsize=12 )
plt.ylabel('predicted RMSE',fontsize=12)
fig.savefig("output_figures/S2.pdf",
       bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
