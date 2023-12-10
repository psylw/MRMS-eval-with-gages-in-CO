# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import rioxarray as rxr
from datetime import timedelta
import sys

sys.path.append('class')

from NRMSD import nrmsd

#%%
#####################################################################################################################   CALCULATE MCE and save

compare = pd.read_feather('output/window_values_new')

# %%
#compare['norm_diff'] = nrmsd(compare)

#compare.to_feather('output/window_values_new')

# %%
#####################################################################################################################   REMOVE SAMPLES WHERE NOTHING HAPPENING FOR MRMS

# remove timesteps where total accum 0
compare = compare.loc[compare.total_mrms_accum>0].reset_index(drop=True)


# %%

#####################################################################################################################   ADD FEATURES

# TIME
compare=compare.rename(columns={"index": "start"})
compare['len_m']=[len(compare.mrms_15_int[i]) for i in range(len(compare))]
compare['end']=[compare['start'][i]+timedelta(minutes=compare.len_m[i].astype('float'))  for i in range(len(compare))]
# %%
# MAX INTENSITY
compare['max_mrms']=[np.max(compare.mrms_15_int[i]) for i in range(len(compare))]
compare['max_gage']=[np.max(compare['15_int'][i]) for i in range(len(compare))]
# %%
# MAX ACCUM
compare['max_accum_atgage']=[pd.DataFrame(compare.mrms_accum[i]).max()[0] for i in range(len(compare))]
# %%
# median, std, mean, var OF POSITIVE MRMS intensity VALUES
median_int=[]
std_int=[]
var_int=[]
mean_int=[]

for i in compare.index:
    intensity = np.array(compare.mrms_15_int[i])
    int_pos = intensity[intensity>0]
    
    median_int.append(np.median(int_pos))
    std_int.append(np.std(int_pos))
    var_int.append(np.var(int_pos))
    mean_int.append(np.mean(int_pos))
    
    
compare['median_int_point']=median_int
compare['std_int_point']=std_int
compare['var_int_point']=var_int
compare['mean_int_point']=mean_int
# %%
# median, std, mean, var OF POSITIVE MRMS accumulation VALUES
median_accum=[]
std_accum=[]
var_accum=[]
mean_accum=[]

for i in compare.index:
    accumulation = np.array(compare.mrms_accum[i])
    accum_pos = accumulation[accumulation>0]
    
    median_accum.append(np.median(accum_pos))
    std_accum.append(np.std(accum_pos))
    var_accum.append(np.var(accum_pos))
    mean_accum.append(np.mean(accum_pos))
    
    
compare['median_accum_point']=median_accum
compare['std_accum_point']=std_accum
compare['var_accum_point']=var_accum
compare['mean_accum_point']=mean_accum
# %%
compare = compare.reset_index()
compare['duration']=[pd.DataFrame(compare.mrms_15_int[i]) for i in range(len(compare))]
compare['duration']=[compare.duration[i].loc[compare.duration[i][0]>0].index[-1]-compare.duration[i].loc[compare.duration[i][0]>0].index[0] for i in range(len(compare))]
# %%

# month
compare['month']=[compare['start'][i].month for i in range(len(compare))]
# hour
compare['hour']=[compare['start'][i].hour for i in range(len(compare))]
#%%
compare = pd.read_feather('output\\window_values_new')

#####################################################################################################################   RESTART AND CONTINUE
compare = pd.read_feather(parentDir+'\\intermediate_datadev')

# bring in RQI
# open RQI for 2021,2022
RQI_folder = os.path.join(parentDir,"MRMS","RQI_cat_yr_CO")
filenames_RQI = glob.glob(RQI_folder+'\\'+'*.grib2')

RQI = xr.open_mfdataset(filenames_RQI,engine = "cfgrib",chunks={'time': '50MB'})

# get times and gage locations
#RQI = RQI.sel(longitude=lon,latitude=lat,drop=True)

RQI = RQI.where(RQI>=0)
RQI = RQI.where(RQI.longitude<=256,drop=True)

RQI = RQI.sel(longitude=lon,latitude=lat,method='nearest',drop=True)
RQI = RQI.drop_duplicates(dim=['latitude','longitude'])

RQI = RQI.sortby(RQI.latitude)
RQI = RQI.sortby(RQI.longitude)


r = []
for i in range(len(compare)):
    # select mrms at coordinate
    r_g = RQI.sel(longitude=compare.longitude[i],latitude=compare.latitude[i],drop=True,method='nearest')
    x = r_g.sel(time=slice(compare.start[i].round('H'),compare.end[i].round('H'))).unknown.values
    r.append(np.min(x[x>=0]))
    print(i/len(compare))
# add to database
compare['RQI']=r
compare['RQI']

#%%

#####################################################################################################################   RESTART AND CONTINUE

compare = pd.read_feather(parentDir+'\\intermediate_datadev')

compare = compare.drop(columns=['index','level_0'])
# %%
elev = '\\CO_SRTM1arcsec__merge.tif'
data_folder = os.path.join('..','..','elev_data')

codtm = rxr.open_rasterio(data_folder+elev)
# change lon to match global lat/lon in grib file
codtm = codtm.assign_coords(x=(((codtm.x + 360))))

codtm = codtm.rename({'x':'longitude','y':'latitude'})

elevation = [codtm.sel(longitude=compare.mrms_lon[i][0],latitude=compare.mrms_lat[i][0],
                       method='nearest').values[0] for i in range(len(compare))]
#elevation = codtm.sel(longitude=lon,latitude=lat,method='nearest')
compare['point_elev']=elevation
#%%
s = '\\CO_SRTM1arcsec_slope.tif'

coslope = rxr.open_rasterio(data_folder+s)

# change lon to match global lat/lon in grib file
coslope = coslope.assign_coords(x=(((coslope.x + 360))))

coslope = coslope.rename({'x':'longitude','y':'latitude'})

#slope = coslope.sel(longitude=lon,latitude=lat,method='nearest')
slope = [coslope.sel(longitude=compare.mrms_lon[i][0],latitude=compare.mrms_lat[i][0],
                     method='nearest').values[0] for i in range(len(compare))]


compare['point_slope']=slope

compare.point_slope=compare.point_slope.where(compare.point_slope.between(0,100))
#%%
# aspect at gage
asp = '\\CO_SRTM1arcsec_aspect.tif'

codtm = rxr.open_rasterio(data_folder+asp)
# change lon to match global lat/lon in grib file
codtm = codtm.assign_coords(x=(((codtm.x + 360))))

codtm = codtm.rename({'x':'longitude','y':'latitude'})

aspect = [codtm.sel(longitude=compare.mrms_lon[i][0],latitude=compare.mrms_lat[i][0],
                       method='nearest').values[0] for i in range(len(compare))]
compare['point_aspect']=aspect
#%%
compare['mrms_lat'] = [compare.mrms_lat[i][0] for i in compare.index]
compare['mrms_lon'] = [compare.mrms_lon[i][0] for i in compare.index]
#####################################################################################################################   SAVE
compare = compare.drop(columns=['level_0', 'index', 'start', 'accum', '15_int', 'gage_lat', 'gage_lon',
       'gage_source', 'mrms_accum', 'mrms_15_int', 
       'total_gage_accum',  'len_m', 'end',
       'max_gage'])
#%%
compare.to_feather('output\\train_test')
# %%