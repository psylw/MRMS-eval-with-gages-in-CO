# %%
import os
import xarray as xr
from dask.distributed import Client
import matplotlib.pyplot as plt
from os import listdir
#import rioxarray as rxr
import glob
import pickle
import pandas as pd
import numpy as np

#####################################################################################################################   IMPORT GAGE DATA

# Load the dictionary from the Pickle file
with open('test//gage_all.pickle', 'rb') as file:
    gage = pickle.load(file)

#client = Client()

#####################################################################################################################   IMPORT MRMS DATA 
# see mrms_atgage.py for how this dataset was built
mrms_folder = os.path.join('mrms_atgage')
filenames_rate = os.listdir(mrms_folder)
#####################################################################################################################   LOOK AT GAGE DATA, LOOK AT MRMS DATA
# MOVE TO ANOTHER FILE

#####################################################################################################################   SAVE GAGE AND MRMS IN 8HR CHUNKS
# %%
target = []
predict = []

for yr in range(2018,2024):
    print(yr)
    name = [s for s in filenames_rate if str(yr) in s][0]
    rate = xr.open_dataset(mrms_folder+'//'+name,chunks={'time': '500MB'})
    rate = rate.where(rate>=0)
    rate = rate*(2/60)

    for i in range(len(gage)):
        for j in range(len(gage[i])):
            if i!=2 and gage[i][j][1]==yr:

                g = gage[i][j][4]
                g['gage_lat'] = gage[i][j][2]
                g['gage_lon'] = gage[i][j][3]
                g['gage_source'] = gage[i][j][0]

                rate_gage = rate.sel(latitude = gage[i][j][2], longitude= gage[i][j][3], method='nearest')
                mrms_lat = rate_gage.latitude.values
                mrms_lon = rate_gage.longitude.values

                rate_gage = rate_gage.to_dataframe()
                rate_gage = rate_gage.unknown.resample('1min').asfreq().fillna(0)
                # select mrms times within gage times, only look at mrms data when gage is recording
                rate_gage = rate_gage.loc[rate_gage.index.isin(g.index)]
                int_gage = rate_gage.rolling(15,min_periods=1).sum()*(60/15)

                mrms = pd.concat([rate_gage.rename('mrms_accum'),int_gage.rename('mrms_15_int')],axis=1)
                mrms['mrms_lat'] = mrms_lat
                mrms['mrms_lon'] = mrms_lon
                # rechunk gage and mrms 
                target.append(g.resample('8h').agg(list))
                predict.append(mrms.resample('8h').agg(list))

            elif i==2 and gage[i][1]==yr:

                g = gage[i][4]
                g['gage_lat'] = gage[i][2]
                g['gage_lon'] = gage[i][3]
                g['gage_source'] = gage[i][0]

                rate_gage = rate.sel(latitude = gage[i][2], longitude= gage[i][3], method='nearest')
                mrms_lat = rate_gage.latitude.values
                mrms_lon = rate_gage.longitude.values

                rate_gage = rate_gage.to_dataframe()
                rate_gage = rate_gage.unknown.resample('1min').asfreq().fillna(0)
                # select mrms times within gage times, only look at mrms data when gage is recording
                rate_gage = rate_gage.loc[rate_gage.index.isin(g.index)]
                int_gage = rate_gage.rolling(15,min_periods=1).sum()*(60/15)

                mrms = pd.concat([rate_gage.rename('mrms_accum'),int_gage.rename('mrms_15_int')],axis=1)
                mrms['mrms_lat'] = mrms_lat
                mrms['mrms_lon'] = mrms_lon
                # rechunk gage and mrms 
                target.append(g.resample('8h').agg(list))
                predict.append(mrms.resample('8h').agg(list))
#%%
target = pd.concat(target)
predict = pd.concat(predict)

compare = pd.concat([target,predict],axis=1)

# %%
#####################################################################################################################   REMOVE SAMPLES WHERE NOTHING HAPPENING AND SAVE
compare['total_gage_accum']=[np.sum(compare.accum[i]) for i in range(len(compare))]

compare['total_mrms_accum']=[np.sum(compare.mrms_accum[i]) for i in range(len(compare))]

# save window values
out = compare.loc[(compare.total_mrms_accum>0)|(compare.total_gage_accum>0)]

out.reset_index().to_feather('output//window_values_new')
# %%
