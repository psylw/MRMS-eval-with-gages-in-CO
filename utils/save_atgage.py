# %%
##################   save mrms data at locations where gage records exist, speed up time series compare   ###################################################################################################
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
filepath_rqi = 'MRMS\\RQI_2min_cat_month_CO\\'
filepath_storm = 'storm_stats\\'

rate = 'rate.nc'
rqi = 'rqi.nc'
storm = 'storm_id.nc'

# %%
for i in range(2023,2024):
    print(i)
    # open rate files for year
    mrms_folder = os.path.join('..','..','..',filepath_rqi)
    filenames_rate = glob.glob(mrms_folder+str(i)+'*.grib2')
    print(filenames_rate)
    rate = xr.open_mfdataset(filenames_rate, chunks={'time': '500MB'})
    # save where gage record exists
    lat = year_lat.get(i)
    lon = year_lon.get(i)
    print(len(np.unique(lat)))    # how many gage locations exist?
    print(len(np.unique(lon)))    # how many gage locations exist?
    rate = rate.sel(longitude=lon,latitude=lat,method='nearest',drop=True)
    rate = rate.drop_duplicates(dim=['latitude','longitude'])
    rate = rate.sortby(rate.latitude)
    rate = rate.sortby(rate.longitude)
    print(len(rate.latitude)) # how many gages exist in same mrms px?
    print(len(rate.longitude)) # how many gages exist in same mrms px?
    # save
    name = str(i)+'_gage_only_'+rqi
    rate.to_netcdf(path=name)
# %%
