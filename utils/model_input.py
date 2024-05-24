# %%
###############################################################################
# clean dataset, manually set cross folds, determine correlated features
###############################################################################

import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def model_input(df):

    df = df.dropna()
    #test = df.loc[df.norm_diff>70]
    #test.groupby(['start','mrms_lat','mrms_lon']).count()
    # this gage has most big outliers, see code above, remove it
    df = df.loc[(df.mrms_lat!=40.57499999999929)&(df.mrms_lon!=254.91499899999639)]
    
    # remove samples where total accumulation less than 1
    df = df.loc[(df.total_mrms_accum>1)].reset_index(drop=True)

    df = df.drop(columns=['start','storm_id'])


    # %%
    # group by location, keep same location in separate train/test and folds
    grouped = df.groupby(['mrms_lat','mrms_lon']).count().total_mrms_accum
    weights = 1.0 / grouped

    for i in range(0,1000):
        test = grouped.sample(frac=.25,weights = weights, random_state=i).reset_index()

        test_lat,test_lon = test.mrms_lat,test.mrms_lon
        test = df.loc[(df.mrms_lat.isin(test_lat))&(df.mrms_lon.isin(test_lon))]

        if len(test)/len(df)>.2:
            break

    #%%
    train = df.loc[~df.index.isin(test.index)]
    print(len(test)/len(df))
    print(len(train)/len(df))
    #%%
    n_splits = 5

    for idx in range(0,1000):
        fold=[]
        split_idx = []
        train = train.reset_index(drop=True)
        sample = train
        for i in range(n_splits):
            test_s = sample.groupby(['mrms_lat','mrms_lon']).count().total_mrms_accum

            weights = 1.0 / grouped

            split = test_s.sample(frac=1/(n_splits-i), weights=weights,random_state=idx).reset_index()

            split_lat,split_lon = split.mrms_lat,split.mrms_lon
            
            split = sample.loc[(sample.mrms_lat.isin(split_lat))&(sample.mrms_lon.isin(split_lon))]
            
            split_idx.append(split.index)
            fold.append(len(split.index)/len(train))
            sample = sample.loc[~sample.index.isin(split.index)]

        if np.std(fold)<.011:
            break
    #%%
    cv = []

    for i in range(n_splits):
        fold_test_idx = split_idx[i]
        fold_train_idx = train.loc[~train.index.isin(fold_test_idx)].index
        cv.append([fold_train_idx,fold_test_idx])

    ######EXP dropping lat/lon
    #test = test.drop(columns=['mrms_lat','mrms_lon'])
    #train = train.drop(columns=['mrms_lat','mrms_lon'])
    # %%
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = (scaler.fit_transform(train.drop(columns=['norm_diff'])),
                                        
    scaler.fit_transform(test.drop(columns=['norm_diff'])),
    train.norm_diff.values,
    test.norm_diff.values)
    # %%
    
    import sys
    sys.path.append('test')
    from feature_select import feature_select
    all_permutations, plot = feature_select(X_train, df,.55)

    return cv, test, train,X_train, X_test, y_train, y_test, all_permutations, plot