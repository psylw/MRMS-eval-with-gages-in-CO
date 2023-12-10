# %%
# create unique storm label for spatially connected regions where precip>0. 
# use class/patch_stormid_acrossmonths.py to patch storm ids across months
import xarray as xr
import numpy as np
import pandas as pd
import skimage.measure
from skimage.morphology import remove_small_objects
import pyarrow.feather as feather
from dask.distributed import Client
import shutil 
import os
import glob
# %%
def find_storms(mrms_ds):
    # create binary ndarray
    a = xr.where(mrms_ds > 0, 1, 0)
    # label objects
    storm_id, count_storms = skimage.measure.label(a, connectivity=1, return_num=True)
    # remove object SMALLER than 5 px
    #storm_id = remove_small_objects(storm_id, 100)
    
    return storm_id

def storm_properties(storm_id):
    object_features = skimage.measure.regionprops(storm_id)
    #area = [objf["area"] for objf in object_features]
    #axis_major_length = [objf["axis_major_length"] for objf in object_features]
    #axis_minor_length = [objf["axis_minor_length"] for objf in object_features]
    centroid = [objf["centroid"] for objf in object_features]
    # round centroid values to index xarray
    centroid = [[round(i[0]),round(i[1]),round(i[2])] for i in centroid]
    # flatten array to get eccentricity??
    #eccentricity = [objf["eccentricity"] for objf in object_features]

    return centroid

# Create a path to the code file
data_folder = os.path.join('..', '..','..','MRMS','2min_rate_cat_month_CO')
filenames = glob.glob(data_folder+'//'+'*.grib2')

destination = os.path.join('..', '..','..','storm_stats')

#client = Client()
# %%
for i in range(len(filenames)):
    print(i)

    month = xr.open_dataset(filenames[i], engine = "cfgrib",chunks={'time': '500MB'})

    month = month.where(month.longitude<=256,drop=True)
    #month = month.unknown.where(month.unknown > 0, drop=True)
    month = month.unknown

    print(month)
    storm_id = find_storms(month)
    centroid = storm_properties(storm_id)

    data = {'centroid_index':centroid}

    storm_param = pd.DataFrame(data = data)

    name = '//'+filenames[0][-22:-6]+'_regionprops'
    storm_param.to_feather(destination+name)

    storm_id = storm_id.astype('float32')
    # create dataset from storm_id
    time = month.time
    latitude = month.latitude 
    longitude = month.longitude

    ds = xr.Dataset(data_vars=dict(storm_id=(["time", "latitude", "longitude"], storm_id),),
        coords=dict(time=time,latitude=latitude,longitude=longitude,))

    name = '//'+filenames[i][-22:-6]+'_storm_id'

    path = destination+name+'.nc'
    ds.to_netcdf(path=path)


# %%
