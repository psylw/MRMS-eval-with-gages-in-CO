# %%
import pandas as pd
from os import listdir
import os
import numpy as np
from datetime import timedelta
#from mrms_gage import gage_storm

#data_folder = os.path.join('..', '..','..','precip_gage')
def get_disdrom(data_folder):
    filenames = listdir(data_folder+'\\csu_disdrometer')
    filenames.sort(reverse=True)
    # first format
    dis1 = pd.read_csv(data_folder +'\\\csu_disdrometer\\'+ filenames[0]).drop(
        ['RainInt','MOR','Sig.Amp','K.E.','dBz','NumPart'],axis=1).drop(0,axis=0)
    dis1['datetime'] = pd.to_datetime(dis1['Date'] + ' ' + dis1['Time'])
    dis1 = dis1.drop(['Date','Time'],axis=1).rename(columns={"RainAcc": "total_accum"}).set_index('datetime')

    # second
    dis2 = [pd.read_csv(data_folder +'\\csu_disdrometer\\'+ filenames[i]) for i in np.arange(1,5,1)]
    dis2 = pd.concat(dis2)
    dis2 = dis2.loc[dis2.Type=='RainAcc'].drop(['Type','Unnamed: 4'],axis=1)
    dis2 = dis2.rename(columns={"Value": "total_accum"})
    dis2['datetime'] = pd.to_datetime(dis2['Date'] + ' ' + dis2['Time'])
    dis2 = dis2.drop(['Date','Time'],axis=1).set_index('datetime')

    disdrom = pd.concat([dis1,dis2], axis=0).sort_index()
    disdrom.index = disdrom.index + pd.Timedelta(hours=7)

    disdrom = disdrom[~disdrom.index.duplicated(keep='first')].resample('1Min').asfreq().fillna(0)
    disdrom = disdrom.iloc[disdrom.index.month.isin(range(5,10))]

    disdrom = disdrom-disdrom.iloc[0]
    disdrom.loc[disdrom.total_accum<0] = 0

    nonzero = disdrom.loc[disdrom.total_accum>0]

    difference = nonzero.total_accum.diff().fillna(0.01)
    #difference = nonzero.total_accum.diff()
    disdrom['accum'] = difference
    disdrom[disdrom.accum < 0] = 0 # recorder has decimal errors?
    disdrom = disdrom.fillna(0)

    disdrom = disdrom[disdrom.loc[disdrom.accum>0].index[0]:disdrom.loc[disdrom.accum>0].index[-1]]

    disdrom['15_int'] = (disdrom.accum.rolling(15,min_periods=1).sum())*(60/15)

    disdrom = disdrom.drop(['total_accum'],axis=1)

    disdrom = ['csu',2021, 40.706725,254.360608,disdrom]

    return disdrom

