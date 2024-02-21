# %%
###############################################################################
# chunk MRMS and gage data into 8-hour time series for MS corrected data. 
###############################################################################
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
with open('..//output//gage_all.pickle', 'rb') as file:
    gage = pickle.load(file)

#client = Client()
#%%
#####################################################################################################################   IMPORT MRMS DATA 
# see mrms_atgage.py for how this dataset was built
mrms_folder = os.path.join('../mrms_atgage')
filenames = os.listdir(mrms_folder)
#####################################################################################################################   LOOK AT GAGE DATA, LOOK AT MRMS DATA
# MOVE TO ANOTHER FILE

#####################################################################################################################   SAVE GAGE AND MRMS IN 8HR CHUNKS
# %%
target = []
predict = []

for yr in range(2021,2023):
    print(yr)
    name_rate = [s for s in filenames if str(yr) in s and 'ms_accum' in s][0]

    rate = xr.open_dataset(mrms_folder+'//'+name_rate,chunks={'time': '500MB'})
    rate = rate.where(rate>=0)
    
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
#%%
compare = pd.concat([target,predict],axis=1)
# %%

compare = compare.reset_index()
# %%
# %%
#####################################################################################################################   REMOVE SAMPLES WHERE NOTHING HAPPENING AND SAVE
compare['total_gage_accum']=[np.sum(compare.accum[i]) for i in compare.index]

compare['total_mrms_accum']=[np.sum(compare.mrms_accum[i]) for i in compare.index]

# save window values
compare = compare.loc[(compare.total_mrms_accum>0)|(compare.total_gage_accum>0)]
#%%

compare['gage_source'] = [compare.gage_source[i][0] for i in compare.index]
compare['gage_lat'] = [compare.gage_lat[i][0] for i in compare.index]
compare['gage_lon'] = [compare.gage_lon[i][0] for i in compare.index]
compare['mrms_lat'] = [compare.mrms_lat[i][0] for i in compare.index]
compare['mrms_lon'] = [compare.mrms_lon[i][0] for i in compare.index]


#%%
#####################################################################################################################   CALCULATE MCE and save
import sys
sys.path.append('../utils')
from RMSE import rmse

compare['norm_diff'] = rmse(compare)

compare.reset_index(drop=True).to_feather('../output/window_values_ms')
# %%
