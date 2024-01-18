# %%
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

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,RandomizedSearchCV

from sklearn.metrics import  mean_absolute_error,r2_score,mean_pinball_loss, mean_squared_error
from model_input import model_input

df = pd.read_feather('output/train_test2')

df['norm_diff'] = np.load('nRMSE.npy')
#df = df.drop(columns=(['median_int_point','mean_int_point','mean_accum_point','median_accum_point']))

cv,test,train,X_train, X_test, y_train, y_test, all_permutations, plot = model_input(df)
# %%
'''max_corr=[]
for i in range(len(all_permutations)):

    pltcorr = df.drop(columns=['start','storm_id','norm_diff']).iloc[:,list(all_permutations[i])]

    correlation_matrix = pltcorr.corr(method='spearman').abs()
    max_corr.append(np.max(correlation_matrix.values[correlation_matrix.values < 1]))

print(np.max(max_corr))'''

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
# RANDOMLY SAMPLE UNCORRELATED FEATURES TO SEE WHICH PERFORMS BEST
SAMPLE_permutations = random.sample(all_permutations, 10)
scores = []

for name, clf in zip(names, classifiers):
    print(name)
    for idx,perm in enumerate(SAMPLE_permutations):
        x = cross_validate(clf,X_train[:,perm],y_train, cv = cv,
                        scoring=['r2','neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'])
        print(idx)
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

        scores.append([name,idx,
                       x['test_r2'].mean(),
                       x['test_r2'].std(),
                       x['test_neg_mean_absolute_error'].mean(),
                       x['test_neg_mean_absolute_error'].std(),
                       x['test_neg_mean_squared_error'].mean(),
                       x['test_neg_mean_squared_error'].std(),                                                                               x['test_neg_root_mean_squared_error'].mean(),
                       x['test_neg_root_mean_squared_error'].std()])

scores = pd.DataFrame(scores,columns=['names','IDX',
                              'test_r2_mean','test_r2_std',
                              'test_neg_mean_absolute_error_mean','test_neg_mean_absolute_error_std',
                              'test_neg_mean_squared_error_mean','test_neg_mean_squared_error_std',
                              'test_neg_root_mean_squared_error_mean','test_neg_root_mean_squared_error_std'])
# %%
# select optimal noncorrelated permutation per model
model_perm = []
for i in names:
    model_perm.append([i,scores.loc[scores.names==i].sort_values('test_neg_mean_absolute_error_mean').IDX.iloc[-1]])

model_perm = pd.DataFrame(data=model_perm,columns=['idx','name'])

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
    idx = model_perm.loc[model_perm.idx==name].name.values[0]
    mod = RandomizedSearchCV(estimator=clf,
                       param_distributions = param,
                       n_iter=15, 
                       scoring=['r2','neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'],
                       refit='neg_mean_absolute_error',
                       cv=cv)

    _ = mod.fit(X_train[:,SAMPLE_permutations[idx]],y_train)  

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

for name, clf in zip(names, classifiers):
    idx = model_perm.loc[model_perm.idx==name].name.values[0]
    f, ax = plt.subplots(figsize=(6, 6))
    clf.fit(X_train[:,SAMPLE_permutations[idx]],y_train)
    '''    mdi_importances = pd.Series(
    clf.feature_importances_, index=feature_names).sort_values(ascending=True)
    mdi_importances.plot.barh()
    plt.show()'''
    ypred = clf.predict(X_test[:,SAMPLE_permutations[idx]])

    print(name)
    print(r2_score(y_test,ypred))
    print(mean_absolute_error(y_test,ypred))

    sns.scatterplot(x=y_test, y=ypred, s=5, color=".15")
    sns.histplot(x=y_test, y=ypred, bins=50, pthresh=.1, cmap="mako")
    sns.kdeplot(x=y_test, y=ypred, levels=5, color="w", linewidths=1)
    plt.xlim(0,25)
    plt.ylim(0,25)
    plt.plot([0,25],[0,25],color='red')
    plt.title(name)
    plt.show()


# %%
from sklearn.inspection import permutation_importance
# compare train/test permutation feature importance for each model, look for consistency and overfitting
param = {'learning_rate': 0.09318794343729946,
 'max_depth': 6,
 'max_features': None,
 'min_samples_leaf': 10,
 'min_samples_split': 5,
 'n_estimators': 77,
 'random_state': 781}

clf = GradientBoostingRegressor(**param)
clf.fit(X_train[:,SAMPLE_permutations[1]], y_train)

train_result = permutation_importance(
    clf, X_train[:,SAMPLE_permutations[1]], y_train, n_repeats=10, random_state=42, n_jobs=2, scoring='neg_mean_absolute_error'
)
test_results = permutation_importance(
    clf, X_test[:,SAMPLE_permutations[1]], y_test, n_repeats=10, random_state=42, n_jobs=2,scoring='neg_mean_absolute_error'
)
sorted_importances_idx = train_result.importances_mean.argsort()
#%%
train_importances = pd.DataFrame(
    train_result.importances[sorted_importances_idx].T,
    columns=train.iloc[:,list(SAMPLE_permutations[1])].columns[sorted_importances_idx],
)
test_importances = pd.DataFrame(
    test_results.importances[sorted_importances_idx].T,
    columns=test.iloc[:,list(SAMPLE_permutations[1])].columns[sorted_importances_idx],
)

for name, importances in zip(["train", "test"], [train_importances, test_importances]):
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title(f"Permutation Importances ({name} set)")
    ax.set_xlabel("Decrease in accuracy score")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.figure.tight_layout()

# %%
