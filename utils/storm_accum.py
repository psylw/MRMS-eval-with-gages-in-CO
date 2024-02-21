#%%
###############################################################################
# calculate storm accumulation spatial and temporal variation
###############################################################################
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd


# get all coordinate files

# get all rate files
mrms_folder = os.path.join('..', '..','..',"MRMS","2min_rate_cat_month_CO")
storm_folder = os.path.join('..', '..','..',"storm_stats")

file_mrms = glob.glob(mrms_folder+'//'+'*.grib2')
file_storm = glob.glob(storm_folder+'//'+'*_coord')
#%%
for year in range(2018,2024):
    print(year)
    for month in ['may','jun','jul','aug','sep']:
        name_month = [s for s in file_mrms if month in s and str(year) in s][0]
        print(name_month)
        m = xr.open_dataset(name_month)
        name_month = [s for s in file_storm if month in s and str(year) in s][0]
        print(name_month)
        s = pd.read_feather(name_month)

        # get mrms into 2min accum from rate
        m = m.where(m.longitude<=256)
        m = m.where(m>=0)
        m = m*(2/60)

        spatial_var_accum = []
        temp_var_accum = []

        i=0
        for storm in s.index:
            print(i/len(s))
            index = s.iloc[storm]
            d = {'time':index.time,'latitude':index.latitude,'longitude':index.longitude}
            temp = pd.DataFrame(data=d)

            temp = temp.groupby(['time','latitude','longitude']).max().to_xarray()

            m_storm = m.sel(time=temp.time,latitude=temp.latitude,longitude=temp.longitude)

            # sum across all coordinates, get temporal variance
            temp_var_accum.append([year, month, index.storm_id,float(m_storm.sum(dim=['latitude','longitude']).unknown.std().values)])

            # sum across all timesteps, get spatial variance
            spatial_var_accum.append([year, month,index.storm_id,float(m_storm.sum(dim=['time']).unknown.std().values)])
            i+=1

        temp_var_accum = pd.DataFrame(temp_var_accum,columns=['year','month','storm_id','temp_var_accum']).fillna(0)

        spatial_var_accum = pd.DataFrame(spatial_var_accum,columns=['year','month','storm_id','spatial_var_accum']).fillna(0)
        
        # save
        spatial_var_accum.to_feather(storm_folder+str(year)+month+'spatial_var_accum')
        temp_var_accum.to_feather(storm_folder+str(year)+month+'temp_var_accum')


# %%
