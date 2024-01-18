# %%
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier, AdaBoostClassifier,BaggingClassifier
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate,RandomizedSearchCV
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
df['nrmsd'] = np.load(os.path.join( 'output', 'nrmsd_mscorrect.npy'))

# %%

df['label']=0
df.loc[df.nrmsd<.1,['label']]=1
# %%
df = df.drop(columns=['gage_id', 'mrms_accum_atgage','gage_accum','onoff', 'mce'])

# %%
pos_frac = len(df.loc[df.label==1])/len(df)
# select gages for testing, weight number of samples so frac positive close to global frac pos
test_s = df.groupby(['latitude','longitude']).count().total_accum_atgage
test_s_pos = df.loc[df.label==1].groupby(['latitude','longitude']).count().total_accum_atgage
test_s_pos_frac = test_s_pos/test_s
# weight locations with frac pos closer to global frac pos higher
weights = 1.0 / np.abs(pos_frac - test_s_pos_frac) 
weights = weights.fillna(.5)

# add loop to get proportion of pos to match
test_s = test_s.sample(frac=.2,weights = weights,random_state=6).reset_index()


test_lat,test_lon = test_s.latitude,test_s.longitude
test = df.loc[(df.latitude.isin(test_lat))&(df.longitude.isin(test_lon))]
train = df.loc[~df.index.isin(test.index)]
print(len(test.loc[test.label==1])/len(test))
print(len(train.loc[train.label==1])/len(train))

# manually set validation folds to separate gages
# do for cv
n_splits = 5
split_idx = []

train = train.reset_index(drop=True)
sample = train

for i in range(n_splits):

    # select gages for testing, weight number of samples so frac positive close to global frac pos
    test_s = sample.groupby(['latitude','longitude']).count().total_accum_atgage
    test_s_pos = sample.loc[sample.label==1].groupby(['latitude','longitude']).count().total_accum_atgage
    test_s_pos_frac = test_s_pos/test_s
    # weight locations with frac pos closer to global frac pos higher
    weights = 1.0 / np.abs(pos_frac - test_s_pos_frac) 
    weights = weights.fillna(5)

    split = test_s.sample(frac=1/(n_splits-i), weights=weights,random_state=4).reset_index()

    split_lat,split_lon = split.latitude,split.longitude
    
    split = sample.loc[(sample.latitude.isin(split_lat))&(sample.longitude.isin(split_lon))]
    
    split_idx.append(split.index)
    print(len(split.loc[split.label==1])/len(split))
    
    sample = sample.loc[~sample.index.isin(split.index)]

folds = []

for i in range(n_splits):
    fold_test_idx = split_idx[i]
    fold_train_idx = train.loc[~train.index.isin(fold_test_idx)].index
    folds.append([fold_train_idx,fold_test_idx])

scaler = StandardScaler()
X_train, X_test, y_train, y_test = (scaler.fit_transform(train.drop(columns=['label','nrmsd'])),
                                    
scaler.fit_transform(test.drop(columns=['label','nrmsd'])),
train.label.values,
test.label.values)

cv = folds # is this indexing correctly????
# %%
names = [
    "Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Bagged Tree",
    "Neural Net",
    "AdaBoost",
    "Logistic Regression",
    "xgboost",
    "Gradient Boosting",
    "SVC"
]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    BaggingClassifier(random_state=0),
    MLPClassifier(random_state=0,max_iter=800),
    AdaBoostClassifier(random_state=0),
    LogisticRegression(random_state=0,max_iter=600),
    xgb.XGBClassifier(random_state=0),
    GradientBoostingClassifier(random_state=0),
    SVC(random_state=0,probability=True)
]
# %%
# baseline
avgp=[]
f1=[]

for name, clf in zip(names, classifiers):
    clf = clf

    x = cross_validate(clf,X_train,y_train, cv = cv,
                     scoring=['f1','average_precision'])

    f1.append([name,str(x['test_f1'].mean())[0:4], str(x['test_f1'].std())[0:6]])
    avgp.append([name,str(x['test_average_precision'].mean())[0:4],str(x['test_average_precision'].std())[0:6]])

f1_baseline = pd.DataFrame(f1,columns=['names','f1','std']).sort_values(['f1'])
avp_baseline = pd.DataFrame(avgp,columns=['names','avgp','std']).sort_values(['avgp'])

avp_baseline
# %%
# initial hyperparameter tuning
classifiers = [
    Pipeline([
        ('sampling', SMOTE(random_state=0)),
        ('clf', KNeighborsClassifier())
    ]),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    Pipeline([
        ('sampling', SMOTE(random_state=0)),
        ('clf', BaggingClassifier(random_state=0))
    ]),
    Pipeline([
        ('sampling', SMOTE(random_state=0)),
        ('clf', MLPClassifier(random_state=0,max_iter=800))
    ]),
    Pipeline([
        ('sampling', SMOTE(random_state=0)),
        ('clf', AdaBoostClassifier(random_state=0))
    ]),
    LogisticRegression(random_state=0,max_iter= 800),
    xgb.XGBClassifier(random_state=0),
    Pipeline([
        ('sampling', SMOTE(random_state=0)),
        ('clf', GradientBoostingClassifier(random_state=0))
    ]),
    SVC(random_state=0,probability=True)
]

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
