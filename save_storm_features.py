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

#####################################################################################################################   OPEN FILE WITH POINT FEATURES

compare = pd.read_feather('output/train_test2')


#####################################################################################################################   IMPORT MONTHLY STORM STATS 

storm_folder = os.path.join('..', '..','..',"storm_stats")
# ACCUMULATION
file_temp = glob.glob(storm_folder+'//'+'*temp_var_accum')
file_spatial = glob.glob(storm_folder+'//'+'*spatial_var_accum')

# VELOCITY
file_vel = glob.glob(storm_folder+'//'+'*velocity')
# AREA
file_area = glob.glob(storm_folder+'//'+'*area')
# %%
# FOR EACH SAMPLE, GET STORM STATS FOR ALL STORMS THAT OCCURED IN THAT SAMPLE
# add float to make storm_ids unique for each month (sample and storm stat)
def find_files_by_year_and_month(year, month, filenames):
    # Convert numeric month to string representation
    month_str = {
        1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
        7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
    }[month]

    # Form the year and month strings
    year_str = str(year)

    # Filter files by matching year and month
    matching_files = [
        filename for filename in filenames
        if year_str in filename and month_str in filename
    ]

    return matching_files

temp_atgage = []
spatial_atgage = []
velocity = []
area = []

for idx in compare.index:
    sample = compare.iloc[idx]
    month = sample.start.month
    year = sample.start.year
    storm_ids = sample.storm_id

    temp = pd.read_feather(find_files_by_year_and_month(year, month, file_temp)[0])
    spatial = pd.read_feather(find_files_by_year_and_month(year, month, file_spatial)[0])
    vel = pd.read_feather(find_files_by_year_and_month(year, month, file_vel)[0])    
    a = pd.read_feather(find_files_by_year_and_month(year, month, file_area)[0])

    temp_atgage.append(temp.loc[temp.storm_id.isin(storm_ids)].temp_var_accum.mean())
    spatial_atgage.append(spatial.loc[spatial.storm_id.isin(storm_ids)].spatial_var_accum.mean())

    velocity.append(vel.loc[vel.storm_id.isin(storm_ids)].velocity.mean())

    area.append(a.loc[a.storm_id.isin(storm_ids)].area.mean())

# %%
compare['temp_var_accum'] = temp_atgage
compare['spatial_var_accum'] = spatial_atgage
compare['velocity'] = velocity
compare['area'] = area
# %%
compare.to_feather('output\\train_test2')
# %%
