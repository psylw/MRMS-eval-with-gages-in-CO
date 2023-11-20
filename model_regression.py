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

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,RandomizedSearchCV

from sklearn.metrics import  mean_absolute_error,r2_score

# %%

# open both window values
file_path = os.path.join('..', '..', 'train_test')
df = pd.read_feather(file_path)

file_path = os.path.join('..', '..', 'min_intensity_gage')
# remove samples where max mrms intensity < min possible gage intensity
min_int = pd.read_feather(file_path)
min_int['gage_id'] = min_int.index
min_int.min_intensity = min_int.min_intensity
df['min_int'] = [min_int.loc[min_int.gage_id==df.gage_id[i][0]].min_intensity.values[0] for i in df.index]

df = df.query('max_mrms > min_int')

#df = df.reset_index(drop=True).drop(columns=['min_int','gage_id','max_accum_atgage'])
df = df.reset_index(drop=True).drop(columns=['min_int','max_accum_atgage'])
df.gage_id = [df.gage_id[i][0] for i in df.index]

# shift lon to 255.5, was 255 when i developed dataset
df = df.loc[df.longitude<255.5]

df = df.dropna()
# %%
df['error'] = np.load(os.path.join( 'output', 'nrmsd_mscorrect_mean.npy'))
#df['error'] = np.load(os.path.join( 'output', 'raw_error.npy'))

# %%
df = df.drop(columns=['gage_id', 'mrms_accum_atgage','gage_accum','onoff', 'mce'])

# %%
grouped = df.groupby(['latitude','longitude']).count().total_accum_atgage
weights = 1.0 / grouped

test = grouped.sample(frac=.25,weights = weights, random_state=0).reset_index()

test_lat,test_lon = test.latitude,test.longitude
test = df.loc[(df.latitude.isin(test_lat))&(df.longitude.isin(test_lon))]
train = df.loc[~df.index.isin(test.index)]
print(len(test)/len(df))
print(len(train)/len(df))

# %%
n_splits = 5
split_idx = []

train = train.reset_index(drop=True)
sample = train

for i in range(n_splits):
    test_s = sample.groupby(['latitude','longitude']).count().total_accum_atgage

    weights = 1.0 / grouped

    split = test_s.sample(frac=1/(n_splits-i), weights=weights,random_state=200).reset_index()

    split_lat,split_lon = split.latitude,split.longitude
    
    split = sample.loc[(sample.latitude.isin(split_lat))&(sample.longitude.isin(split_lon))]
    
    split_idx.append(split.index)
    print(len(split.index)/len(train))
    sample = sample.loc[~sample.index.isin(split.index)]
#%%
for i in range(n_splits):
    fold_test_idx = split_idx[i]

    x1,y1 = train.iloc[fold_test_idx].longitude,train.iloc[fold_test_idx].latitude

    plt.scatter(x1,y1,label=str(i))
    
    plt.legend()

plt.scatter(test.longitude,test.latitude,label='test')
#%%
cv = []

for i in range(n_splits):
    fold_test_idx = split_idx[i]
    fold_train_idx = train.loc[~train.index.isin(fold_test_idx)].index
    cv.append([fold_train_idx,fold_test_idx])

# %%
scaler = StandardScaler()
X_train, X_test, y_train, y_test = (scaler.fit_transform(train.drop(columns=['error'])),
                                    
scaler.fit_transform(test.drop(columns=['error'])),
train.error.values,
test.error.values)


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
    MLPRegressor(random_state=0,max_iter=800),
    AdaBoostRegressor(random_state=0),

    xgb.XGBRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
    SVR()
]
# %%
feature_names = train.drop(columns='error').columns

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
    plt.xlim(0,25)
    plt.ylim(0,25)
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
    r2.append([name,x['test_score'].mean(),x['test_score'].std()])

r2 = pd.DataFrame(r2,columns=['names','r2','std']).sort_values(['r2'])
# %%
#xgb
param = {"learning_rate": [0.5, 0.25, 0.1, 0.05, 0.01,.001], 
                            "n_estimators": [64, 100, 200,400,600],
                            "max_depth":[5,10,20,30],
                                'min_child_weight': [1, 5, 10],
                                'gamma': [1, 1.5, 2, 5,6],
                                'subsample': [0.6, 0.8, 1.0],
                                'colsample_bytree': [0.6, 0.8, 1.0],
                                'lambda':np.arange(0,10,.3),
                                'alpha': np.arange(0,1,.1)}

clf = xgb.XGBRegressor(random_state=0)

mod = RandomizedSearchCV(estimator=clf,
                    param_distributions = param,
                    n_iter=10, 
                    scoring='neg_mean_absolute_error',
                    refit="neg_mean_absolute_error",
                    cv=cv)

_ = mod.fit(X_train,y_train)  

# %%
param = {'subsample': 0.6,
 'n_estimators': 400,
 'min_child_weight': 5,
 'max_depth': 30,
 'learning_rate': 0.01,
 'lambda': 0.8999999999999999,
 'gamma': 6,
 'colsample_bytree': 1.0,
 'alpha': 0.7000000000000001}

clf = xgb.XGBRegressor(random_state=0, **param)

clf.fit(X_train,y_train)

print(r2_score(y_test,clf.predict(X_test)))

feature_names = train.drop(columns='error').columns
mdi_importances = pd.Series(
    clf.feature_importances_, index=feature_names).sort_values(ascending=True)
mdi_importances.plot.barh()
plt.show()

#%%
# initial hyperparameter tuning

param = [
    # knn
    {'clf__n_neighbors': [2,3,5,8,10], 
     'clf__weights': ['uniform', 'distance'], 'clf__leaf_size':[10,20,30,50] },
    # dt
    {'max_depth': [5,10,20,30],'min_samples_split':[2,4,6,8],'min_samples_leaf':[1,2,4,6],
     'class_weight':['balanced',{0: 1.0, 1: 10.0},{0: 1.0, 1: 20.0},{0: 1.0, 1: 50.0}],
     'criterion':['gini','entropy','log_loss']},
    # RF
    {"n_estimators": [64, 100, 200,400], 
     'class_weight':['balanced',{0: 1.0, 1: 10.0},{0: 1.0, 1: 20.0},{0: 1.0, 1: 50.0}],
                                 "max_depth":[5,10,20,30],
                                          "min_samples_split":range(2,9),
                                         "min_samples_leaf":range(1,8)},
    # bagging
    {"clf__n_estimators": [64, 100, 200,400,600],'clf__max_samples':[1,3,5,8],'clf__max_features':[1,3,5,8]},
    # MLP
    {"clf__hidden_layer_sizes": [50,100,200,400], 
                                "clf__activation": ['identity', 'logistic', 'tanh', 'relu'],
                                          "clf__learning_rate_init":np.arange(0.0001,0.002,0.0003)},
    #Ada
    {'clf__n_estimators': [10,20,50,100], 'clf__learning_rate':[.01,.1,1,2]},
    
    # logistic
    {
     'class_weight':['balanced',{0: 1.0, 1: 10.0},{0: 1.0, 1: 20.0},{0: 1.0, 1: 50.0}]},
    #xgb
    {"learning_rate": [0.5, 0.25, 0.1, 0.05, 0.01,.001], 
                                "n_estimators": [64, 100, 200,400,600],
                                "max_depth":[5,10,20,30],
                                 'min_child_weight': [1, 5, 10],
                                 'gamma': [1, 1.5, 2, 5,6],
                                 'subsample': [0.6, 0.8, 1.0],
                                 'colsample_bytree': [0.6, 0.8, 1.0],
                                'scale_pos_weight':[10,20,30,50]},
    #gbc
    {"clf__learning_rate": [1, 0.5, 0.25, 0.1, 0.05, 0.01], 
                                "clf__n_estimators": [64, 100, 200,400],
                                "clf__max_depth":range(1,11),
                                          "clf__min_samples_split":range(2,8),
                                          "clf__min_samples_leaf":range(1,8)},
    # SVC
    {'C': [0.1, 1, 10, 100],'class_weight':['balanced',{0: 1.0, 1: 10.0},{0: 1.0, 1: 20.0},{0: 1.0, 1: 50.0}],
     'kernel':['linear', 'poly', 'rbf', 'sigmoid']} 
]

hyp = []
for name, clf, param in zip(names, classifiers, param):

    clf = clf

    mod = RandomizedSearchCV(estimator=clf,
                       param_distributions = param,
                       n_iter=10, 
                       scoring=['average_precision',"f1",'precision','recall'],
                       refit="average_precision",
                       cv=cv)

    _ = mod.fit(X_train,y_train)  

    hyp.append([name, 
                pd.DataFrame(mod.cv_results_).sort_values('mean_test_average_precision',
                                                          ascending=False).iloc[0].params,
                               pd.DataFrame(mod.cv_results_).sort_values('mean_test_average_precision',
                                                          ascending=False).iloc[0].mean_test_average_precision,
                                              pd.DataFrame(mod.cv_results_).sort_values('mean_test_average_precision',
                                                          ascending=False).iloc[0].mean_test_f1,
               pd.DataFrame(mod.cv_results_).sort_values('mean_test_average_precision',
                                                          ascending=False).iloc[0].mean_test_precision,
               pd.DataFrame(mod.cv_results_).sort_values('mean_test_average_precision',
                                                          ascending=False).iloc[0].mean_test_recall])
# %%
