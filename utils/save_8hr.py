# %%
###############################################################################
# chunk MRMS and gage data into 8-hour time series. Includes storm ids and RQI. Calculate RMSE for each time series.
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

for yr in range(2018,2024):
    print(yr)
    name_rate = [s for s in filenames if str(yr) in s and 'rate' in s][0]
    name_rqi = [s for s in filenames if str(yr) in s and 'rqi' in s][0]
    name_storm = [s for s in filenames if str(yr) in s and 'storm' in s][0]
    
    rate = xr.open_dataset(mrms_folder+'//'+name_rate,chunks={'time': '500MB'})
    rate = rate.where(rate>=0)
    rate = rate*(2/60)

    rqi = xr.open_dataset(mrms_folder+'//'+name_rqi,chunks={'time': '500MB'})
    rqi = rqi.where(rqi>=0)

    storm = xr.open_dataset(mrms_folder+'//'+name_storm,chunks={'time': '500MB'})
    storm = storm.where(storm.storm_id>0)
    
    for i in range(len(gage)):
        for j in range(len(gage[i])):
            if i!=2 and gage[i][j][1]==yr:

                g = gage[i][j][4]
                g['gage_lat'] = gage[i][j][2]
                g['gage_lon'] = gage[i][j][3]
                g['gage_source'] = gage[i][j][0]

                rate_gage = rate.sel(latitude = gage[i][j][2], longitude= gage[i][j][3], method='nearest')
                
                storm_gage = storm.sel(latitude = gage[i][j][2], longitude= gage[i][j][3], method='nearest')
                storm_gage = storm_gage.storm_id.to_dataframe()
                
                mrms_lat = rate_gage.latitude.values
                mrms_lon = rate_gage.longitude.values

                rate_gage = rate_gage.to_dataframe()
                rate_gage = rate_gage.unknown.resample('1min').asfreq().fillna(0)
                
                # select mrms times within gage times, only look at mrms data when gage is recording
                rate_gage = rate_gage.loc[rate_gage.index.isin(g.index)]
                int_gage = rate_gage.rolling(15,min_periods=1).sum()*(60/15)

                storm_gage = storm_gage.loc[storm_gage.index.isin(g.index)]
                
                mrms = pd.concat([rate_gage.rename('mrms_accum'),int_gage.rename('mrms_15_int'),storm_gage.storm_id.rename('storm_id')],axis=1)

                mrms['mrms_lat'] = mrms_lat
                mrms['mrms_lon'] = mrms_lon
                # rechunk gage and mrms 
                target.append(g.resample('8h').agg(list))
                
                rqi_gage = rqi.sel(latitude = gage[i][j][2], longitude= gage[i][j][3], method='nearest')
                rqi_gage = rqi_gage.to_dataframe()
                # select mrms times within gage times, only look at mrms data when gage is recording
                rqi_gage = rqi_gage.loc[rqi_gage.index.isin(g.index)].unknown

                mrms['rqi'] = rqi_gage

                predict.append(mrms.resample('8h').agg(list))

            elif i==2 and gage[i][1]==yr:

                g = gage[i][4]
                g['gage_lat'] = gage[i][2]
                g['gage_lon'] = gage[i][3]
                g['gage_source'] = gage[i][0]

                rate_gage = rate.sel(latitude = gage[i][2], longitude= gage[i][3], method='nearest')
                storm_gage = storm.sel(latitude = gage[i][2], longitude= gage[i][3], method='nearest')
                storm_gage = storm_gage.storm_id.to_dataframe()

                mrms_lat = rate_gage.latitude.values
                mrms_lon = rate_gage.longitude.values

                rate_gage = rate_gage.to_dataframe()
                rate_gage = rate_gage.unknown.resample('1min').asfreq().fillna(0)
                # select mrms times within gage times, only look at mrms data when gage is recording
                rate_gage = rate_gage.loc[rate_gage.index.isin(g.index)]
                int_gage = rate_gage.rolling(15,min_periods=1).sum()*(60/15)

                storm_gage = storm_gage.loc[storm_gage.index.isin(g.index)]

                mrms = pd.concat([rate_gage.rename('mrms_accum'),int_gage.rename('mrms_15_int'),storm_gage.storm_id.rename('storm_id')],axis=1)

                mrms['mrms_lat'] = mrms_lat
                mrms['mrms_lon'] = mrms_lon
                # rechunk gage and mrms 
                target.append(g.resample('8h').agg(list))

                rqi_gage = rqi.sel(latitude = gage[i][2], longitude= gage[i][3], method='nearest')
                rqi_gage = rqi_gage.to_dataframe()
                
                # select mrms times within gage times, only look at mrms data when gage is recording
                rqi_gage = rqi_gage.loc[rqi_gage.index.isin(g.index)].unknown

                mrms['rqi'] = rqi_gage
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
#fix lat lon and rqi
# fix storm_id, save only unique values 
compare['storm_id'] = [pd.Series(compare.storm_id[i]).dropna().unique() for i in compare.index] #drop na
#%%
compare['gage_source'] = [compare.gage_source[i][0] for i in compare.index]
compare['gage_lat'] = [compare.gage_lat[i][0] for i in compare.index]
compare['gage_lon'] = [compare.gage_lon[i][0] for i in compare.index]
compare['mrms_lat'] = [compare.mrms_lat[i][0] for i in compare.index]
compare['mrms_lon'] = [compare.mrms_lon[i][0] for i in compare.index]




# %%
rqi_mean = []
rqi_median = []
rqi_min = []
rqi_max = []
rqi_std = []
for i in compare.index:
    rqi = compare.rqi[i]
    rqi = pd.DataFrame(rqi).dropna()
    rqi_mean.append(rqi.mean().values[0])
    rqi_median.append(rqi.median().values[0])
    rqi_min.append(rqi.min().values[0])
    rqi_max.append(rqi.max().values[0])
    rqi_std.append(rqi.std().values[0])
# %%
compare['rqi_mean'] = rqi_mean
compare['rqi_median'] = rqi_median
compare['rqi_min'] = rqi_min
compare['rqi_max'] = rqi_max
compare['rqi_std'] = rqi_std
#%%
compare = compare.drop(columns = 'rqi')
#%%
#####################################################################################################################   CALCULATE MCE and save
import sys
sys.path.append('../utils')
from RMSE import rmse

compare['norm_diff'] = rmse(compare)

compare.reset_index(drop=True).to_feather('../output/window_values_new')
# %%
