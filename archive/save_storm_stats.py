#%%
import xarray as xr
import numpy as np
import skimage.measure
import pandas as pd
from scipy.spatial import distance
import skimage.measure
from skimage.morphology import remove_small_objects,closing,binary_closing
from scipy import ndimage
from datetime import datetime,timedelta
import os
from dask.distributed import Client
import glob
from shapely.geometry import MultiPoint
import geopandas as gpd
import cartopy.crs as ccrs

mrms_folder = os.path.join('..', '..','..',"MRMS","2min_rate_cat_month_CO")
storm_folder = os.path.join('..', '..','..',"storm_stats")

file_mrms = glob.glob(mrms_folder+'//'+'*.grib2')
file_storm = glob.glob(storm_folder+'//'+'*.nc')

#client = Client()
#%%
for year in range(2018,2024):
    print(year)
    for month in ['may','jun','jul','aug','sep']:
        print(month)
        name_month = [s for s in file_mrms if month in s and str(year) in s][0]
        print(name_month)
        m = xr.open_dataset(name_month,chunks={'time': '500MB'})
        name_month = [s for s in file_storm if month in s and str(year) in s][0]
        print(name_month)
        s = xr.open_dataset(name_month,chunks={'time': '500MB'})

        # get mrms into 2min accum from rate
        m = m.where(m.longitude<=256,drop=True)
        m = m.where(m>=0)
        m = m*(2/60)

        # get storm ids
        storms = s.storm_id.values
        print(storms)
        storms = storms[~np.isnan(storms)]
        storms = storms[storms>0]
        storms = np.unique(storms)
        
        # split into sections to fit into memory
        sample = np.array_split(storms,20)
        
        # spatial variability of intensity/accum--get max/total of intensity/accum for entire storm footprint, calc variability
        spatial_var_int = []
        spatial_var_accum = []

        # temporal variability of intensity/accum--get max/total of intensity/accum for each timestep, calc variability
        temp_var_int = []
        temp_var_accum = []

        i=0
        for storm in sample:
            print(i/len(sample))
            sample_storms = s.where(s.storm_id.isin(storm),drop=True)
            sample_mrms = m.sel(time=sample_storms.time,latitude = sample_storms.latitude, longitude = sample_storms.longitude)
            sample_mrms = sample_mrms.assign({"storm_id": sample_storms.storm_id})
            
            sample_mrms = sample_mrms.to_dataframe().rename(columns={'unknown': 'accum'})

            sample_mrms['int'] = sample_mrms.accum.rolling(15,min_periods=1).sum()*(60/15)

            space = sample_mrms.drop(['step','heightAboveSea','valid_time'],
                                            axis=1).reset_index().groupby(['latitude','longitude','storm_id'])
            
            time = sample_mrms.drop(['step','heightAboveSea','valid_time'],
                                axis=1).reset_index().groupby(['time','storm_id'])
            
            accum_space = space.sum().accum.reset_index()
            accum_time = time.sum().accum.reset_index()
            int_space = space.max().int.reset_index()
            int_time = time.max().int.reset_index()
            a = []
            b = []
            c = []
            d = []
            for i in accum_space.storm_id.unique():
                a.append([i,accum_space.loc[accum_space.storm_id==i].accum.std()])
                b.append([i,accum_time.loc[accum_time.storm_id==i].accum.std()])
                c.append([i,int_space.loc[int_space.storm_id==i].int.std()])
                d.append([i,int_time.loc[int_time.storm_id==i].int.std()])                

            spatial_var_accum.append(pd.DataFrame(a, columns=['storm_id','spatial_var_accum']).fillna(0))
            temp_var_accum.append(pd.DataFrame(b, columns=['storm_id','temp_var_accum']).fillna(0))
            spatial_var_int.append(pd.DataFrame(c, columns=['storm_id','spatial_var_int']).fillna(0))
            temp_var_int.append(pd.DataFrame(d, columns=['storm_id','temp_var_int']).fillna(0))
            i+=1
    spatial_var_accum = pd.concat(spatial_var_accum).reset_index()
    temp_var_accum = pd.concat(temp_var_accum).reset_index()
    spatial_var_int = pd.concat(spatial_var_int).reset_index()
    temp_var_int = pd.concat(temp_var_int).reset_index()

    # save
    name = ['spatial_var_accum','temp_var_accum','spatial_var_int','temp_var_int']

    spatial_var_accum.to_feather(storm_folder+str(year)+month+name[0])
    temp_var_accum.to_feather(storm_folder+str(year)+month+name[1])
    spatial_var_int.to_feather(storm_folder+str(year)+month+name[2])  
    temp_var_int.to_feather(storm_folder+str(year)+month+name[3])
            
    # %%
