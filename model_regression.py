# %%
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,RandomizedSearchCV

from sklearn.metrics import  mean_absolute_error,r2_score

# %%

# open both window values
#file_path = os.path.join('..', '..', 'train_test')
file_path = os.path.join( 'output', 'train_test')
df = pd.read_feather(file_path)
#%%
df = df.dropna()

#  %%
# add rmse
test = pd.read_feather('output\window_values_new')
test = test.loc[test.total_mrms_accum>0].reset_index(drop=True)
(df.total_mrms_accum-test.total_mrms_accum).max()
df[['rqi_mean', 'rqi_median', 'rqi_min', 'rqi_max',
       'rqi_std', 'norm_diff']] = test[['rqi_mean', 'rqi_median', 'rqi_min', 'rqi_max',
       'rqi_std', 'norm_diff']]

# %%

#df = df.loc[(df.total_mrms_accum>1)].reset_index(drop=True)

df['norm_diff'] = df.norm_diff/df.max_mrms
df = df.loc[(df.total_mrms_accum>1)].reset_index(drop=True)
# remove correlated features...do this in different file 
# %%
df = df.drop(columns=['max_mrms',
       'max_accum_atgage',  'std_int_point',
       'var_int_point', 'mean_int_point', 'median_accum_point',
       'std_accum_point', 'var_accum_point', 'mean_accum_point'])

# %%
grouped = df.groupby(['mrms_lat','mrms_lon']).count().total_mrms_accum
weights = 1.0 / grouped

test = grouped.sample(frac=.25,weights = weights, random_state=0).reset_index()

test_lat,test_lon = test.mrms_lat,test.mrms_lon
test = df.loc[(df.mrms_lat.isin(test_lat))&(df.mrms_lon.isin(test_lon))]
train = df.loc[~df.index.isin(test.index)]
print(len(test)/len(df))
print(len(train)/len(df))

# %%
n_splits = 5
split_idx = []

train = train.reset_index(drop=True)
sample = train

for i in range(n_splits):
    test_s = sample.groupby(['mrms_lat','mrms_lon']).count().total_mrms_accum

    weights = 1.0 / grouped

    split = test_s.sample(frac=1/(n_splits-i), weights=weights,random_state=200).reset_index()

    split_lat,split_lon = split.mrms_lat,split.mrms_lon
    
    split = sample.loc[(sample.mrms_lat.isin(split_lat))&(sample.mrms_lon.isin(split_lon))]
    
    split_idx.append(split.index)
    print(len(split.index)/len(train))
    sample = sample.loc[~sample.index.isin(split.index)]
#%%
'''for i in range(n_splits):
    fold_test_idx = split_idx[i]

    x1,y1 = train.iloc[fold_test_idx].mrms_lon,train.iloc[fold_test_idx].mrms_lat

    plt.scatter(x1,y1,label=str(i))
    
    plt.legend()

plt.scatter(test.mrms_lon,test.mrms_lat,label='test')'''

#%%
cv = []

for i in range(n_splits):
    fold_test_idx = split_idx[i]
    fold_train_idx = train.loc[~train.index.isin(fold_test_idx)].index
    cv.append([fold_train_idx,fold_test_idx])

# %%
scaler = StandardScaler()
X_train, X_test, y_train, y_test = (scaler.fit_transform(train.drop(columns=['norm_diff'])),
                                    
scaler.fit_transform(test.drop(columns=['norm_diff'])),
train.norm_diff.values,
test.norm_diff.values)


# %%
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
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(random_state=0),
    BaggingRegressor(random_state=0),
    MLPRegressor(random_state=0),
    AdaBoostRegressor(random_state=0),

    xgb.XGBRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
    SVR()
]
# %%
feature_names = train.drop(columns='norm_diff').columns

for name, clf in zip(names, classifiers):
    clf.fit(X_train,y_train)
    '''    mdi_importances = pd.Series(
    clf.feature_importances_, index=feature_names).sort_values(ascending=True)
    mdi_importances.plot.barh()
    plt.show()'''
    ypred = clf.predict(X_test)

    print(name)
    print(r2_score(y_test,ypred))
    print(mean_absolute_error(y_test,ypred))

    plt.scatter(y_test,ypred)
    #plt.xlim(0,25)
    #plt.ylim(0,25)
    plt.plot([0,25],[0,25],color='red')
    plt.title(name)
    plt.show()

# %%
# baseline
r2=[]

for name, clf in zip(names, classifiers):
    print(name)
    x = cross_validate(clf,X_train,y_train, cv = cv,
                     scoring='r2')
    print(x['test_score'])
    r2.append([name,x['test_score'].mean(),x['test_score'].std()])

r2 = pd.DataFrame(r2,columns=['names','r2','std']).sort_values(['r2'])


#%%
# initial hyperparameter tuning
from hyperparam import param
param = param
hyp = []
for name, clf, param in zip(names, classifiers, param):
    print(name)
    clf = clf

    mod = RandomizedSearchCV(estimator=clf,
                       param_distributions = param,
                       n_iter=15, 
                       scoring='neg_mean_absolute_error',
                       refit='neg_mean_absolute_error',
                       cv=cv)

    _ = mod.fit(X_train,y_train)  


    print(pd.DataFrame(mod.cv_results_).sort_values('rank_test_score').params.iloc[0])
    hyp.append(pd.DataFrame(mod.cv_results_).sort_values('rank_test_score').params.iloc[0])


# %%
param = [#Nearest Neighbors
{'leaf_size': 55, 'n_neighbors': 34, 'p': 2, 'weights': 'uniform'},
#Decision Tree
{'max_depth': 13, 'min_samples_leaf': 8, 'min_samples_split': 8},

#Random Forest
{'bootstrap': False, 'max_depth': 32, 'max_features': 'sqrt', 'min_samples_leaf': 10, 'min_samples_split': 8, 'n_estimators': 76},
#Bagged Tree
{'max_features': 0.8556277572937148, 'max_samples': 0.6537113949138937, 'n_estimators': 54},
#Neural Net
{'activation': 'tanh', 'alpha': 0.008705799756697682, 'hidden_layer_sizes': 120, 'learning_rate': 'invscaling', 'learning_rate_init': 0.03574510193293592, 'max_iter': 926, 'solver': 'sgd'},
#AdaBoost
{'learning_rate': 0.10744518516445699, 'n_estimators': 298},

#xgboost
{'colsample_bytree': 0.9897619999843497, 'gamma': 0.2895269462694678, 'learning_rate': 0.025705445295337102, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 315, 'reg_alpha': 0.1980621551139078, 'reg_lambda': 0.8096303523179117, 'subsample': 0.6598150064654535},

#Gradient Boosting
{'learning_rate': 0.0838805794537557, 'max_depth': 6, 'max_features': 'log2', 'min_samples_leaf': 13, 'min_samples_split': 18, 'n_estimators': 91},

#SVC
{'C': 6.402095318891977, 'degree': 5, 'epsilon': 0.14055757662783375, 'gamma': 'scale', 'kernel': 'linear'}

]
# %%
param = hyp
# add parameters to model
classifiers = [
    KNeighborsRegressor(**param[0]),
    DecisionTreeRegressor(random_state=0,**param[1]),
    RandomForestRegressor(random_state=0,**param[2]),
    BaggingRegressor(random_state=0,**param[3]),
    MLPRegressor(random_state=0,**param[4]),
    AdaBoostRegressor(random_state=0,**param[5]),

    xgb.XGBRegressor(random_state=0,**param[6]),
    GradientBoostingRegressor(random_state=0,**param[7]),
    SVR(**param[8])
]
# %%
