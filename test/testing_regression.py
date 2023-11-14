
import numpy as np
import pandas as pd
from numpy import nan

from sklearn.ensemble import BaggingRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.model_selection import train_test_split,cross_validate,cross_val_predict,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,auc
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score,precision_recall_curve,accuracy_score,roc_curve

import xgboost as xgb
import os

from datetime import datetime,timedelta

codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)

df = pd.read_feather(parentDir+'\\train_test')

# recalculate mce bc i replaced neg values with zero
compare = pd.read_feather('window_values')

# compute MSE from chunks
current_datetime = datetime.now()

mce = []
for i in compare.index:
    g = compare.gage[i]
    m = compare.mrms[i]    
    
    datetime_g = [current_datetime + timedelta(minutes=i) for i in range(len(g))]
    datetime_m = [current_datetime + timedelta(minutes=i) for i in range(len(m))]

    # segment into 10 min chunks
    g = np.sort(pd.DataFrame(data=g,index=datetime_g).resample('10min').max().values)
    m = np.sort(pd.DataFrame(data=m,index=datetime_m).resample('10min').max().values)
    
    mce.append(1-(np.mean(np.abs(m - g))/np.mean(np.abs(g - np.mean(g)))))


# remove samples where max mrms intensity < min possible gage intensity
min_int = pd.read_feather(parentDir+'\\min_intensity_gage')
min_int['gage_id'] = min_int.index
min_int.min_intensity = min_int.min_intensity
df['min_int'] = [min_int.loc[min_int.gage_id==df.gage_id[i][0]].min_intensity.values[0] for i in df.index]

df = df.query('max_mrms > min_int')

#df = df.reset_index(drop=True).drop(columns=['min_int','gage_id','max_accum_atgage'])
df = df.reset_index(drop=True).drop(columns=['min_int','max_accum_atgage'])
df.gage_id = [df.gage_id[i][0] for i in df.index]

# remove multiple gages in same MRMS grid
test = df.groupby(['latitude','longitude']).agg(list)
test['num_gage'] = [len(np.unique(test.gage_id[i])) for i in test.index]
test['i_gage'] = [np.unique(test.gage_id[i]) for i in test.index]

dup = test.loc[test.num_gage>1].reset_index().i_gage
gage_dup = [dup[i][1] for i in dup.index]
df = df.loc[~df.gage_id.isin(gage_dup)]

# shift lon to 255.5, was 255 when i developed dataset
df = df.loc[df.longitude<255.5]

df = df.drop(columns=['gage_id','mrms_accum_atgage','gage_accum']).dropna()

# remove -inf
df.loc[df.mce<-10000,['mce']]=-1

# regression 
test_s = df.groupby(['latitude','longitude']).count().total_accum_atgage
weights = test_s/len(test_s)

test_s = test_s.sample(frac=.2,weights = weights).reset_index()
test_lat,test_lon = test_s.latitude,test_s.longitude

test = df.loc[(df.latitude.isin(test_lat))&(df.longitude.isin(test_lon))]
train = df.loc[~df.index.isin(test.index)]

scaler = StandardScaler()
X_train, X_test, y_train, y_test = (scaler.fit_transform(train.drop(columns=['mce'])),
scaler.fit_transform(test.drop(columns=['mce'])),
train.mce.values,
test.mce.values)

cv = KFold(n_splits=5, shuffle=False)



names = [
    "Nearest Neighbors",
    #"RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Bagged Tree",
    "Neural Net",
    "AdaBoost",
    #"Logistic Regression",
    "xgboost",
    "Gradient Boosting",
    "SVC"
]

classifiers = [
    KNeighborsRegressor(),
    #SVC(gamma='auto',random_state=0),
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(random_state=0),
    BaggingRegressor(random_state=0),
    MLPRegressor(random_state=0),
    AdaBoostRegressor(random_state=0),
    #LogisticRegression(random_state=0,class_weight='balanced'),
    xgb.XGBRegressor(random_state=0),
    GradientBoostingRegressor(random_state=0),
    SVR()
]

for name, clf in zip(names, classifiers):
    clf = clf
    
    x = cross_validate(clf,X_train,y_train, cv = cv,
                     scoring=['r2'])
    
    print(x['test_r2'])
