#%%
###############################################################################
# calculate storm area and velocity
###############################################################################

import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
# open libraries
import os
import xarray as xr
from dask.distributed import Client
import matplotlib.pyplot as plt
from os import listdir
import rioxarray as rxr
import glob
#import seaborn as sns
from shapely.geometry import MultiPoint
import geopandas as gpd
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import gc


# get all coordinate files
storm_folder = os.path.join('..', '..','..',"storm_stats")

file_storm = glob.glob(storm_folder+'//'+'*_coord')
file_storm = file_storm
#####################################################################################################################   CALCULATE VELOCITY FOR EACH STORM, DO THIS OUTSIDE

#%%
#for year in range(2018,2024):
for year in range(2019,2020):
    print(year)
    #for month in ['may','jun','jul','aug','sep']:
    for month in ['aug','sep']:
        name_month = [s for s in file_storm if month in s and str(year) in s][0]
        print(name_month)
        s = pd.read_feather(name_month)

        velocity = []
        area = []

        i=0
        for storm in s.index:
            print(i/len(s))
            index = s.iloc[storm]
            d = {'time':index.time,'latitude':index.latitude,'longitude':index.longitude}
            storm = pd.DataFrame(data=d)
            storm['fill'] = 1
            temp = storm.groupby(['latitude','longitude']).max().to_xarray()

            # Assuming 'mask' contains the binary mask representing the object of interest
            mask = temp['fill']  # Replace 'mask' with your variable name

            # Count the number of cells (which represent 1 km each) that belong to the object
            num_cells = (mask == 1).sum()  # Assuming the object is represented by 1 in the mask
            area.append([year, month,index.storm_id,float(num_cells)])

            x = storm.longitude.values
            y = storm.latitude.values

            points = gpd.GeoSeries.from_xy(x, y, crs="EPSG:4326")
            crs = ccrs.LambertConformal(central_latitude=38.5, central_longitude=-105)
            points = points.to_crs(crs)
            d = {'points':points}
            points = gpd.GeoDataFrame(d)
            storm = pd.concat([storm,points],axis=1)

            # get centroid for each timestep
            storm = storm.reset_index().groupby('time').agg(list)
            storm['centroid'] = [MultiPoint(storm.points[i]).centroid for i in storm.index]
            centroid = gpd.GeoDataFrame(geometry=storm.centroid)

            distance=centroid.distance(centroid.shift(1))
            
            velocity.append([year, month,index.storm_id,(distance/(2*60)).mean()])    
            i+=1
    
        velocity = pd.DataFrame(velocity,columns=['year','month','storm_id','velocity']).fillna(0)
        area = pd.DataFrame(area,columns=['year','month','storm_id','area']).fillna(0)

        # save
        velocity.to_feather(storm_folder+str(year)+month+'velocity')
        area.to_feather(storm_folder+str(year)+month+'area')
    





# %%
