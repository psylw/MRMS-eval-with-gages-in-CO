

# min/max values
# time conversion
# rounding issues with 15min int?
# stuck gages

# %%
import os
import xarray as xr
from dask.distributed import Client
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
#import rioxarray as rxr
import glob
import pickle
from datetime import datetime, timedelta

#####################################################################################################################   IMPORT GAGE DATA

data_folder = os.path.join('..','output')

with open('gage_all.pickle', 'rb') as file:
    gage = pickle.load(file)


# %%
# look at min/max
min_accum = []
max_accum = []
min_int = []
max_int = []

for i in range(len(gage)):
    for j in range(len(gage[i])):
        if i!=2:
            min_accum.append(gage[i][j][4].accum.min())
            max_accum.append(gage[i][j][4].accum.max())
            min_int.append(gage[i][j][4]['15_int'].min())
            max_int.append(gage[i][j][4]['15_int'].max())       

gage[2][4].accum.min()
gage[2][4].accum.max()
gage[2][4]['15_int'].max()
gage[2][4]['15_int'].min()

        
# %%
# open MRMS
# open rate files for year
mrms_folder = os.path.join('..','mrms_atgage')
filenames_rate = os.listdir(mrms_folder)

for yr in range(2018,2024):
    name = [s for s in filenames_rate if str(yr) in s][0]
    rate = xr.open_dataset(mrms_folder+'//'+name,chunks={'time': '500MB'})
    rate = rate.where(rate>=0)
    rate = rate*(2/60)
    # look at all values per gage year

    for i in range(len(gage)):
        for j in range(len(gage[i])):
            if i!=2 and gage[i][j][1]==yr:
                rate_gage = rate.sel(latitude = gage[i][j][2], longitude= gage[i][j][3], method='nearest',drop=True)
                
                rate_gage.to_dataframe().unknown.cumsum().plot(label='mrms')
                gage[i][j][4].accum.cumsum().plot(label='gage')
                
                plt.title(gage[i][j][0]+', '+str(gage[i][j][1])+', '+str(gage[i][j][2])+', '+str(gage[i][j][3])+', ')
                plt.legend()
                plt.show()
# %%
name = [s for s in filenames_rate if str(2021) in s][0]
rate = xr.open_dataset(mrms_folder+'//'+name,chunks={'time': '500MB'})
rate = rate.where(rate>=0)
rate = rate*(2/60)

rate_gage = rate.sel(latitude = gage[2][2], longitude= gage[2][3], method='nearest',drop=True)
rate_gage.to_dataframe().unknown.cumsum().plot(label='mrms')
gage[2][4].accum.cumsum().plot(label='gage')
plt.title(gage[2][0]+', '+str(gage[2][1])+', '+str(gage[2][2])+', '+str(gage[2][3])+', ')
plt.legend()
plt.show()

# compare max int recorded by gage to mrms
rate_gage = rate_gage.to_dataframe()

int_gage = rate_gage.unknown.resample('1min').asfreq()
int_gage = int_gage.fillna(0)
int_gage = int_gage.rolling(15,min_periods=1).sum()*(60/15)

iddx = gage[2][4]['15_int'].sort_values().index[-1]
start = iddx -timedelta(hours=6)
end = iddx+ timedelta(hours=6)

int_gage[start:end].plot(label='mrms')
gage[2][4]['15_int'][start:end].plot(label='gage')
plt.legend()

# %%
# open MRMS
# open rate files for year
mrms_folder = os.path.join('..','mrms_atgage')
filenames_rate = os.listdir(mrms_folder)

for yr in range(2018,2024):
    name = [s for s in filenames_rate if str(yr) in s][0]
    rate = xr.open_dataset(mrms_folder+'//'+name,chunks={'time': '500MB'})
    rate = rate.where(rate>=0)
    rate = rate*(2/60)
    
    # look at all values per gage year

    for i in range(len(gage)):
        for j in range(len(gage[i])):
            if i!=2 and gage[i][j][1]==yr:
                rate_gage = rate.sel(latitude = gage[i][j][2], longitude= gage[i][j][3], method='nearest',drop=True)
                rate_gage = rate_gage.to_dataframe()
                int_gage = rate_gage.unknown.resample('1min').asfreq()
                int_gage = int_gage.fillna(0)
                int_gage = int_gage.rolling(15,min_periods=1).sum()*(60/15)
                
                start = gage[i][j][4]['15_int'].idxmax()- timedelta(hours=6)
                end = gage[i][j][4]['15_int'].idxmax()+ timedelta(hours=6)

                int_gage[start:end].plot(label='mrms')
                gage[i][j][4]['15_int'][start:end].plot(label='gage')

                
                plt.title(gage[i][j][0]+', '+str(gage[i][j][1])+', '+str(gage[i][j][2])+', '+str(gage[i][j][3])+', ')
                plt.legend()
                plt.show()