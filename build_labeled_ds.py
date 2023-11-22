import os
import xarray as xr
from dask.distributed import Client
import matplotlib.pyplot as plt
from os import listdir
#import rioxarray as rxr
import glob
import pickle

#####################################################################################################################   IMPORT GAGE DATA

# Load the dictionary from the Pickle file
with open('output//data.pickle', 'rb') as file:
    gage = pickle.load(file)

# get list of keys (lat/lon)
coord = [i for i in gage.keys()]


# Create a path to the code file
codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)

client = Client()

#####################################################################################################################   IMPORT MRMS DATA 
# see mrms_atgage.py for how this dataset was built

# 2 min ACCUMULATION for months may thru sep, 2021/2022, in MDT
#m = xr.open_dataset(parentDir+'\\'+'mrms_atgage_stormsonly.nc',chunks={'time': '50MB'})
m = xr.open_dataset(parentDir+'\\'+'mrms_atgage_stormsonly_radaronly.nc',chunks={'time': '50MB'})

#####################################################################################################################   SAVE GAGE AND MRMS IN 8HR CHUNKS
# select mrms 1 gage at a time, find storms
target = []
predict = []
for j in years
open mfdataset for 5 months
for i in range(len(coord)):
    
    print('gagecount='+str(i))
    # get corresponding mrms coordinate
    latt = coord[i][0]
    lont = coord[i][1]
    # select mrms at coordinate
    m_g = m.sel(longitude=lont,latitude=latt,method='nearest',drop=True)
    # convert to pandas dataframe, speeds things up
    m_g = m_g.dropna(dim='time').to_dataframe()
    
    # calculate 15-min intensity
    mrms_15_21 = m_g.iloc[m_g.index.year==2021].resample('1min').asfreq()
    mrms_15_21.unknown = mrms_15_21.unknown.fillna(0)
    mrms_15_21.unknown = (mrms_15_21.unknown.rolling(15,min_periods=1).sum())*(60/15) 
    mrms_15_22 = m_g.iloc[m_g.index.year==2022].resample('1min').asfreq()
    mrms_15_22.unknown = mrms_15_22.unknown.fillna(0)
    mrms_15_22.unknown = (mrms_15_22.unknown.rolling(15,min_periods=1).sum())*(60/15) 

    mrms_15 = pd.concat([mrms_15_21,mrms_15_22],axis=0)
    
    # get gage
    g = gage[coord[i]].fillna(0)

    # 15-min intensity caused weird rounding issues, fix by setting to zero
    g.loc[g['15_int']<0.0001] = 0
    

    # select times after first positive and before last negative, check that gage actually recording
    g = g[g.loc[g['15_int']>0].index[0]:g.loc[g['15_int']>0].index[-1]]

    # select gage data within mrms times, only look at gage data when I have mrms data
    g = g.loc[g.index.isin(mrms_15.index)]
    g['gage']=i
    
    # select mrms times within gage times, only look at mrms data when gage is recording
    mrms_15 = mrms_15.loc[mrms_15.index.isin(g.index)]
    
    mrms_15['accum']=m_g.loc[m_g.index.isin(mrms_15.index)].unknown
    mrms_15.accum = mrms_15.accum.fillna(0)
    
    # rechunk gage and mrms 
    target.append(g.resample('8h').agg(list))
    predict.append(mrms_15.resample('8h').agg(list))
    
target = pd.concat(target)
predict = pd.concat(predict)

d = {'mrms':predict.unknown,'gage':target['15_int'],'storm_id':predict.storm_id,'gage_id':target.gage,
     'mrms_accum_atgage':predict.accum, 'gage_accum':target.accum}
compare = pd.DataFrame(data=d)

compare.storm_id = [np.unique(compare.storm_id[i]) for i in range(len(compare))]
compare.gage_id = [np.unique(compare.gage_id[i]) for i in range(len(compare))]

compare = compare.reset_index()

#####################################################################################################################   CALCULATE MCE

# add downsampled, unsorted
# compute MSE from chunks
current_datetime = datetime.now()

mce_unsorted = []
for i in test.index:
    g = test.gage[i]
    m = test.mrms[i]    

    datetime_g = [current_datetime + timedelta(minutes=i) for i in range(len(g))]
    datetime_m = [current_datetime + timedelta(minutes=i) for i in range(len(m))]

    # segment into 10 min chunks
    g = pd.DataFrame(data=g,index=datetime_g)
    m = pd.DataFrame(data=m,index=datetime_m)
    g_r = g.resample('10min').max().values
    m_r = m.resample('10min').max().values
 
    mce_unsorted.append(1-(np.mean(np.abs(m_r - g_r))/np.mean(np.abs(g_r - np.mean(g_r)))))

test['mce_unsorted'] = mce_unsorted

test.loc[test.mce_unsorted<0,['mce_unsorted']]=-.1

#####################################################################################################################   REMOVE SAMPLES WHERE NOTHING HAPPENING AND SAVE
compare['total_accum_atgage']=[np.sum(compare.mrms_accum_atgage[i]) for i in range(len(compare))]

compare['total_gage_accum']=[np.sum(compare.gage_accum[i]) for i in range(len(compare))]

# save window values
out = compare.loc[(compare.total_accum_atgage>0)|(compare.total_gage_accum>0)]
out.reset_index().to_feather('output//window_values')

#####################################################################################################################   REMOVE SAMPLES WHERE NOTHING HAPPENING FOR MRMS
# remove empty timesteps (empty data)
compare = compare.reset_index(drop=True)
compare['len_m']=[len(compare.mrms[i]) for i in range(len(compare))]
compare = compare.loc[compare.len_m>0].reset_index(drop=True)

# remove timesteps where total accum 0
compare = compare.loc[compare.total_accum_atgage>0].reset_index(drop=True)

compare=compare.rename(columns={"index": "start"})

# get rid of nan storm ids
compare.storm_id = [compare.storm_id[i][~np.isnan(compare.storm_id[i])] for i in range(len(compare))]

# remove samples where no storm_ids exist
compare['len_storm_id']=[len(compare.storm_id[i]) for i in range(len(compare))]
compare = compare.loc[compare.len_storm_id>0]

compare = compare.reset_index(drop=True)

#####################################################################################################################   ADD FEATURES

# TIME
compare['end']=[compare['start'][i]+timedelta(minutes=compare.len_m[i].astype('float'))  for i in range(len(compare))]

# MAX INTENSITY
compare['max_mrms']=[np.max(compare.mrms[i]) for i in range(len(compare))]
compare['max_gage']=[np.max(compare.gage[i]) for i in range(len(compare))]

# MAX ACCUM
compare['max_accum_atgage']=[pd.DataFrame(compare.mrms_accum_atgage[i]).max()[0] for i in range(len(compare))]

# median, std, mean, var OF POSITIVE MRMS intensity VALUES
median_int=[]
std_int=[]
var_int=[]
mean_int=[]

for i in compare.index:
    intensity = np.array(compare.mrms[i])
    int_pos = intensity[intensity>0]
    
    median_int.append(np.median(int_pos))
    std_int.append(np.std(int_pos))
    var_int.append(np.var(int_pos))
    mean_int.append(np.mean(int_pos))
    
    
compare['median_int_point']=median_int
compare['std_int_point']=std_int
compare['var_int_point']=var_int
compare['mean_int_point']=mean_int

# median, std, mean, var OF POSITIVE MRMS accumulation VALUES
median_accum=[]
std_accum=[]
var_accum=[]
mean_accum=[]

for i in compare.index:
    accumulation = np.array(compare.mrms_accum_atgage[i])
    accum_pos = accumulation[accumulation>0]
    
    median_accum.append(np.median(accum_pos))
    std_accum.append(np.std(accum_pos))
    var_accum.append(np.var(accum_pos))
    mean_accum.append(np.mean(accum_pos))
    
    
compare['median_accum_point']=median_accum
compare['std_accum_point']=std_accum
compare['var_accum_point']=var_accum
compare['mean_accum_point']=mean_accum

compare = compare.reset_index()
compare['duration']=[pd.DataFrame(compare.mrms[i]) for i in range(len(compare))]
compare['duration']=[compare.duration[i].loc[compare.duration[i][0]>0].index[-1]-compare.duration[i].loc[compare.duration[i][0]>0].index[0] for i in range(len(compare))]

lat_m = []
lon_m = []
for i in range(len(coord)):
    # get corresponding mrms coordinate
    latt = coord[i][0]
    lont = coord[i][1]
    # select mrms at coordinate
    m_g = m.sel(longitude=lont,latitude=latt,method='nearest')
    lat_m.append(m_g.latitude.values)
    lon_m.append(m_g.longitude.values)
    
# add point data to dataset
# lat/lon
compare['latitude']=[float(lat_m[compare.gage_id.iloc[i][0]]) for i in range(len(compare))]
compare['longitude']=[float(lon_m[compare.gage_id.iloc[i][0]]) for i in range(len(compare))]

# month
compare['month']=[compare['start'][i].month for i in range(len(compare))]
# hour
compare['hour']=[compare['start'][i].hour for i in range(len(compare))]

#####################################################################################################################   SAVE INTERMEDIATE
compare = compare.drop(columns=['mrms','gage','mrms_accum_atgage','gage_accum'])

name = 'intermediate_datadev'
output = parentDir+name
compare.to_feather(output)

#####################################################################################################################   RESTART AND CONTINUE
compare = pd.read_feather(parentDir+'\\intermediate_datadev')

lat = compare.latitude.unique()
lon = compare.longitude.unique()

# multisensor correction
# open multisensor qpe for 2021,2022
mrms_multi_folder = os.path.join(parentDir,"MRMS","1hr_QPE_multi_cat_yr_CO")
filenames_multi = glob.glob(mrms_multi_folder+'\\'+'*.grib2')
mrms_multi = xr.open_mfdataset(filenames_multi,engine = "cfgrib",chunks={'time': '50MB'})
mrms_multi = mrms_multi.where(mrms_multi>=0)

mrms_multi = mrms_multi.where(mrms_multi.longitude<=256,drop=True)

mrms_multi = mrms_multi.sel(longitude=lon,latitude=lat,method='nearest',drop=True)
mrms_multi = mrms_multi.drop_duplicates(dim=['latitude','longitude'])
# open radaronly qpe for 2021,2022
mrms_radar_folder = os.path.join(parentDir,"MRMS","1hr_QPE_radar_cat_yr_CO")
filenames_radar = glob.glob(mrms_radar_folder+'\\'+'*.grib2')
mrms_radar = xr.open_mfdataset(filenames_radar,engine = "cfgrib",chunks={'time': '50MB'})
mrms_radar = mrms_radar.where(mrms_radar>=0)
mrms_radar = mrms_radar.where(mrms_radar.longitude<=256,drop=True)

mrms_radar = mrms_radar.sel(longitude=lon,latitude=lat,method='nearest',drop=True)
mrms_radar = mrms_radar.drop_duplicates(dim=['latitude','longitude'])

correction = (mrms_multi/mrms_radar)
correction = correction.where(correction.unknown != np.inf).fillna(1)

# select only gage locations and storm times
#correction = correction.sel(longitude=lon,latitude=lat,drop=True)

correction = time_change(correction)

correction = correction.sortby(correction.latitude)
correction = correction.sortby(correction.longitude)

c = []
for i in range(len(compare)):
    # select mrms at coordinate
    c_g = correction.sel(longitude=compare.longitude[i],latitude=compare.latitude[i],method='nearest',drop=True)
    
    c.append(c_g.sel(time=slice(compare.start[i].round('H'),compare.end[i].round('H'))).unknown.max().compute().values)
    print(i/len(compare))
# add to database
compare['mult_correct']=c

compare.mult_correct = compare.mult_correct.astype('float')
#####################################################################################################################   SAVE INTERMEDIATE
name = 'intermediate_datadev'
output = parentDir+name
compare.to_feather(output)

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

# change to MST
RQI = time_change(RQI)

r = []
for i in range(len(compare)):
    # select mrms at coordinate
    r_g = RQI.sel(longitude=compare.longitude[i],latitude=compare.latitude[i],drop=True,method='nearest')
    x = r_g.sel(time=slice(compare.start[i].round('H'),compare.end[i].round('H'))).unknown.values
    r.append(np.min(x[x>=0]))
    print(i/len(compare))
# add to database
compare['RQI']=r

#####################################################################################################################   SAVE INTERMEDIATE
name = 'intermediate_datadev'
output = parentDir+name
compare.to_feather(output)

#####################################################################################################################   RESTART AND CONTINUE

compare = pd.read_feather(parentDir+'\\intermediate_datadev')

compare = compare.drop(columns=['index','level_0'])

elev = '\\CO_SRTM1arcsec__merge.tif'
folder = parentDir
elev = folder+elev
codtm = rxr.open_rasterio(elev)
# change lon to match global lat/lon in grib file
codtm = codtm.assign_coords(x=(((codtm.x + 360))))

codtm = codtm.rename({'x':'longitude','y':'latitude'})

elevation = [codtm.sel(longitude=compare.longitude[i],latitude=compare.latitude[i],
                       method='nearest').values[0] for i in range(len(compare))]
#elevation = codtm.sel(longitude=lon,latitude=lat,method='nearest')

s = '\\CO_SRTM1arcsec_slope.tif'
s = folder+s
coslope = rxr.open_rasterio(s)
# change lon to match global lat/lon in grib file
coslope = coslope.assign_coords(x=(((coslope.x + 360))))

coslope = coslope.rename({'x':'longitude','y':'latitude'})

#slope = coslope.sel(longitude=lon,latitude=lat,method='nearest')
slope = [coslope.sel(longitude=compare.longitude[i],latitude=compare.latitude[i],
                     method='nearest').values[0] for i in range(len(compare))]

compare['point_elev']=elevation
compare['point_slope']=slope

compare.point_slope=compare.point_slope.where(compare.point_slope.between(0,100))

storm_id = []
for i in range(len(compare)):
    try:
        storm_id.append(eval(compare.storm_id[i]))
    except:
        test = compare.storm_id[i].split()
        
        test = str(test).replace("[","")
        test = str(test).replace("]","")
        test = [eval(idx) for idx in test.split(', ')]
        test = [float(i) for i in test if len(i)>0]
        storm_id.append(test)
        
compare['storm_id'] = storm_id

# open, select value based on storm_id
months = ['may','jun','jul','aug','sep']
years = [2021,2022]

# open storm_ids
precip_folder = os.path.join(parentDir,"precip_stats")
# add float to make storm_ids unique for each month 
unique = np.arange(0,0.1,0.01)

# max, mean, median, variance
filenames_stats = [i+'_'+str(j)+'_precip_stats' for i in months for j in years]

stats = []
for i in range(len(filenames_stats)):
    s = pd.read_feather(precip_folder+'\\'+filenames_stats[i])
    s.storm_id = s.storm_id+unique[i]
    stats.append(s)
stats = pd.concat(stats)

stats = [stats.loc[stats.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))] ############not sure if this should be max or mean
stats = pd.concat(stats,axis=1).T
stats = stats.drop(columns=['storm_id'])
stats = stats.reset_index()
compare = pd.concat([compare,stats],axis=1)

# elevation at centroid
filenames_elev = [i+'_'+str(j)+'_elev1' for i in months for j in years]

elev = []
for i in range(len(filenames_elev)):
    s = pd.read_feather(precip_folder+'\\'+filenames_elev[i])
    s.storm_id = s.storm_id+unique[i]
    elev.append(s)
elev = pd.concat(elev)
elev = [elev.loc[elev.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
elev = pd.concat(elev,axis=1).T.reset_index()
compare['storm_elev'] = elev.elevation

# average elevation of footprint
# average slope of footprint

filenames_se = [i+'_'+str(j)+'_slope_elev' for i in months for j in years]
se=[]
for i in range(len(filenames_se)):
    s = pd.read_feather(precip_folder+'\\'+filenames_se[i])
    s.storm_id = s.storm_id+unique[i]
    se.append(s)
slope_elev = pd.concat(se)
slope_elev = slope_elev.rename(columns={"elevation": "elevation_foot", "slope": "slope_foot"})
#slope_elev['slope_foot'] = np.where(slope_elev['slope_foot'] <0, 0, slope_elev['slope_foot'])

slope_elev = [slope_elev.loc[slope_elev.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))] ############not sure if this should be max or mean
slope_elev = pd.concat(slope_elev,axis=1).T
slope_elev = slope_elev.drop(columns=['storm_id'])
slope_elev = slope_elev.reset_index()

compare = pd.concat([compare,slope_elev],axis=1)

# aspect at gage
asp = '\\CO_SRTM1arcsec_aspect.tif'
folder = parentDir
asp = folder+asp
codtm = rxr.open_rasterio(asp)
# change lon to match global lat/lon in grib file
codtm = codtm.assign_coords(x=(((codtm.x + 360))))

codtm = codtm.rename({'x':'longitude','y':'latitude'})

aspect = [codtm.sel(longitude=compare.longitude[i],latitude=compare.latitude[i],
                       method='nearest').values[0] for i in range(len(compare))]
compare['point_aspect']=aspect

# aspect storm
# average elevation of footprint
# average slope of footprint

filenames_asp = [i+'_'+str(j)+'_aspect' for i in months for j in years]
asp=[]
for i in range(len(filenames_asp)):
    s = pd.read_feather(precip_folder+'\\'+filenames_asp[i])
    s.storm_id = s.storm_id+unique[i]
    asp.append(s)
aspect = pd.concat(asp)

aspect = [aspect.loc[aspect.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
aspect = pd.concat(aspect,axis=1).T.reset_index()
compare['storm_aspect'] = aspect.aspect


# accum
filenames_accum = [i+'_'+str(j)+'_accum' for i in months for j in years]

accum_mean = []
accum_max = []
accum_median = []
accum_std = []
accum_var = []


for i in range(len(filenames_accum)):
    s = pd.read_feather(precip_folder+'\\'+filenames_accum[i])
    s.storm_id = s.storm_id+unique[i]
    accum_mean.append(s.groupby(['storm_id']).mean())
    accum_max.append(s.groupby(['storm_id']).max())
    accum_median.append(s.groupby(['storm_id']).median())
    accum_std.append(s.groupby(['storm_id']).std())
    accum_var.append(s.groupby(['storm_id']).var())
    

accum_mean = pd.concat(accum_mean).reset_index()
accum_max = pd.concat(accum_max).reset_index()
accum_median = pd.concat(accum_median).reset_index()
accum_std = pd.concat(accum_std).reset_index()
accum_var = pd.concat(accum_var).reset_index()

accum_mean = [accum_mean.loc[accum_mean.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
accum_mean = pd.concat(accum_mean,axis=1).T.reset_index()
compare['accum_mean_storm'] = accum_mean.unknown

accum_max = [accum_max.loc[accum_max.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
accum_max = pd.concat(accum_max,axis=1).T.reset_index()
compare['accum_max_storm'] = accum_max.unknown


accum_median = [accum_median.loc[accum_median.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
accum_median = pd.concat(accum_median,axis=1).T.reset_index()
compare['accum_median_storm'] = accum_median.unknown

accum_std = [accum_std.loc[accum_std.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
accum_std = pd.concat(accum_std,axis=1).T.reset_index()
compare['accum_std_storm'] = accum_std.unknown

accum_var = [accum_var.loc[accum_var.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
accum_var = pd.concat(accum_var,axis=1).T.reset_index()
compare['accum_var_storm'] = accum_var.unknown

# storm duration
filenames_dur = [i+'_'+str(j)+'_dur' for i in months for j in years]

dur = []
for i in range(len(filenames_dur)):
    s = pd.read_feather(precip_folder+'\\'+filenames_dur[i])
    s.storm_id = s.storm_id+unique[i]
    dur.append(s)

dur = pd.concat(dur)
dur = [dur.loc[dur.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
dur = pd.concat(dur,axis=1).T.reset_index()
compare['duration_storm'] = dur['duration (min)']

# major length
# minor length
# eccentricity
# area
# max, mean, median, variance
filenames_shape = [i+'_'+str(j)+'_shape' for i in months for j in years]

shape = []
for i in range(len(filenames_shape)):
    s = pd.read_feather(precip_folder+'\\'+filenames_shape[i])
    s.storm_id = s.storm_id+unique[i]
    shape.append(s)
shape = pd.concat(shape)
shape = [shape.loc[shape.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
shape = pd.concat(shape,axis=1).T.reset_index()
shape = shape.drop(columns=['storm_id']).reset_index()
compare = pd.concat([compare,shape],axis=1)


#####################################################################################################################   SAVE INTERMEDIATE
name = 'intermediate_datadev'
output = parentDir+name
compare.to_feather(output)

#####################################################################################################################   RESTART AND CONTINUE

#####################################################################################################################   CALCULATE VELOCITY FOR EACH STORM, DO THIS OUTSIDE

filenames_coord = [i+'_'+str(j)+'_coord' for i in months for j in years]

coord = []
for i in range(len(filenames_coord)):
    s = pd.read_feather(precip_folder+'\\'+filenames_coord[i])
    s.storm_id = s.storm_id+unique[i]
    coord.append(s)

coord = pd.concat(coord)

compare = pd.read_feather(parentDir+'\\intermediate_datadev2')

# get all unique storms from samples
sid = [compare.storm_id[i] for i in compare.index]
sid = np.concatenate(sid)
sid = np.unique(sid)

coord = coord.loc[coord.storm_id.isin(sid)]

coord = coord.reset_index()

#save sample
name = 'temp_coord'
output = parentDir+'\\'+name
coord.to_pickle(output)

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

codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)

# open sample
coord = pd.read_pickle(parentDir+'\\'+'temp_coord')

sample = np.array_split(coord,20)

del(coord)
gc.collect()

for j in range(len(sample)):
    index = sample[j].reset_index()
    velocity=[]
    
    for i in index.index:
        d = {'time':index.time[i],'latitude':index.latitude[i],'longitude':index.longitude[i]}
        storm = pd.DataFrame(data=d)

        x = storm.longitude.values
        y = storm.latitude.values

        points = gpd.GeoSeries.from_xy(x, y, crs="EPSG:4326")
        crs = ccrs.LambertConformal(central_latitude=38.5, central_longitude=-105)
        points = points.to_crs(crs)
        d = {'points':points}
        points = gpd.GeoDataFrame(d)
        storm = pd.concat([storm,points],axis=1)
        
        test = len(storm.time.unique())
        
        if test>500:
            sample_storm = np.array_split(storm.time.unique(),100)
            c = []
            for k in range(len(sample_storm)):
                s_sample = storm.loc[storm.time.isin(sample_storm[k])]
                # get centroid for each timestep
                s_sample = s_sample.reset_index().groupby('time').agg(list)
                s_sample['centroid'] = [MultiPoint(s_sample.points[i]).centroid for i in s_sample.index]
                centroid = gpd.GeoDataFrame(geometry=s_sample.centroid)
                c.append(centroid)
            c = pd.concat(c)
            distance=c.distance(c.shift(1))
            velocity.append([index.storm_id[i],(distance/(2*60)).max()])    
            print(i/len(index))

        else:
            # get centroid for each timestep
            storm = storm.reset_index().groupby('time').agg(list)
            storm['centroid'] = [MultiPoint(storm.points[i]).centroid for i in storm.index]
            centroid = gpd.GeoDataFrame(geometry=storm.centroid)

            distance=centroid.distance(centroid.shift(1))
            velocity.append([index.storm_id[i],(distance/(2*60)).max()])    
            print(i/len(index))
        
        del(storm)
        del(points)
        del(centroid)
        del(distance)
        gc.collect()
    
    velocity = pd.DataFrame(data={'storm_id':[velocity[i][0] for i in range(len(velocity))],
                   'velocity':[velocity[i][1] for i in range(len(velocity))]})
    # save
    name = 'temp_max_velocity_'+str(j)
    output = parentDir+'\\'+name
    velocity.to_feather(output)
    
    del(velocity)
    gc.collect()

filenames_vel = ['temp_max_velocity_'+str(i) for i in range(20)]

velocity = []
for i in range(len(filenames_vel)):
    s = pd.read_feather(parentDir+'\\'+filenames_vel[i])
    velocity.append(s)

velocity = pd.concat(velocity)

# assign mean velocity to each sample
vel = [velocity.loc[velocity.storm_id.isin(compare.storm_id[i])].mean() for i in range(len(compare))]
vel = pd.concat(vel,axis=1).T.reset_index()
#compare['velocity'] = velocity['velocity']

compare['velocity'] = vel['velocity']


#####################################################################################################################   SAVE
compare = compare.drop(columns=['start', 'mrms', 'gage','storm_id', 'max_gage', 
       'len_m','end', 'len_storm_id', 'total_gage_accum'])

compare.to_feather(parentDir+'\\train_test')