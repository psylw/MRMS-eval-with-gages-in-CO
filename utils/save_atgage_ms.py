# %%
###############################################################################
# save mrms data at locations where gage records exist, speed up time series eval
###############################################################################

import os
import xarray as xr
from dask.distributed import Client
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
#import rioxarray as rxr
import glob
import pickle

##################   IMPORT GAGE DATA   ###################################################################################################
#%%
with open('test//gage_all.pickle', 'rb') as file:
    gage = pickle.load(file)

# %%
# save gage coordinates by year

data_list = []
for i in range(len(gage)):
    if i!=2:
        for j in range(len(gage[i])):
            data_list.append(gage[i][j][1:4])
    else:
        data_list.append(gage[i][1:4])

year_lat = {}
year_lon = {}
for entry in data_list:
    year = entry[0]
    lat = entry[1]
    lon = entry[2]
    
    if year not in year_lat:
        year_lat[year] = [lat]
        year_lon[year] = [lon]
    else:
        year_lat[year].append(lat)
        year_lon[year].append(lon)
#%%
filepath_rate = 'MRMS\\2min_rate_cat_month_CO\\'
filepath_ms = 'MRMS\\1hr_QPE_multi_cat_yr_CO\\'
filepath_radar = 'MRMS\\1hr_QPE_radar_cat_yr_CO\\'

# %%
for i in range(2021,2023):
    print(i)
    rate_folder = os.path.join('..','..','..',filepath_rate)
    ms_folder = os.path.join('..','..','..',filepath_ms)
    radar_folder = os.path.join('..','..','..',filepath_radar)
    # open rate files for year
    filenames_rate = glob.glob(rate_folder+str(i)+'*.grib2')
    filenames_ms = glob.glob(ms_folder+str(i)+'*.grib2')
    filenames_radar = glob.glob(radar_folder+str(i)+'*.grib2')

    rate = xr.open_mfdataset(filenames_rate, chunks={'time': '500MB'})
    ms = xr.open_mfdataset(filenames_ms, chunks={'time': '500MB'})
    radar = xr.open_mfdataset(filenames_radar, chunks={'time': '500MB'})

    correction = (ms/radar)
    correction = correction.where(correction.unknown != np.inf).fillna(1)
    correction = correction.resample(time='2min').pad()

    ms_accum = rate*(2/60)
    ms_accum = ms_accum*correction

    # save where gage record exists
    lat = year_lat.get(i)
    lon = year_lon.get(i)
    print(len(np.unique(lat)))    # how many gage locations exist?
    print(len(np.unique(lon)))    # how many gage locations exist?


    ms_accum = ms_accum.sel(longitude=lon,latitude=lat,method='nearest',drop=True)
    ms_accum = ms_accum.drop_duplicates(dim=['latitude','longitude'])
    ms_accum = ms_accum.sortby(ms_accum.latitude)
    ms_accum = ms_accum.sortby(ms_accum.longitude)


    print(len(ms_accum.latitude)) # how many gages exist in same mrms px?
    print(len(ms_accum.longitude)) # how many gages exist in same mrms px?
    # save
    name = str(i)+'_gage_only_'+'ms_accum.nc'
    ms_accum.to_netcdf(path=name)
# %%
