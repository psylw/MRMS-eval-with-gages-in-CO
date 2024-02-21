#%%
###############################################################################
# resample MRMS data to make size manageable
###############################################################################
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd

# get all rate files
mrms_folder = os.path.join('..', '..','..',"MRMS","2min_rate_cat_month_CO")
rqi_folder = os.path.join('..', '..','..',"MRMS","RQI_2min_cat_month_CO")
storm_folder = os.path.join('..', '..','..',"storm_stats")

file_mrms = glob.glob(mrms_folder+'//'+'*.grib2')
file_rqi = glob.glob(rqi_folder+'//'+'*.grib2')
file_storm = glob.glob(storm_folder+'//'+'*.nc')

#%%
for year in range(2018,2024):
    print(year)
    for month in ['may','jun','jul','aug','sep']:
        ########### OPEN BY YEAR/MONTH
        name_month = [s for s in file_mrms if month in s and str(year) in s][0]
        print(name_month)
        m = xr.open_dataset(name_month, chunks={'time': '500MB'})
        name_month = [s for s in file_rqi if month in s and str(year) in s][0]
        print(name_month)
        r = xr.open_dataset(name_month, chunks={'time': '500MB'})
        name_month = [s for s in file_storm if month in s and str(year) in s][0]
        print(name_month)
        s = xr.open_dataset(name_month, chunks={'time': '500MB'})
        
        ########### CONVERT NANS DEFINE AOI
        m = m.where(m.longitude<=256,drop=True)
        m = m.where(m>=0)
        m = m*(2/60)# get mrms into 2min accum from rate

        r = r.where(r.longitude<=256,drop=True)
        
        s = s.where(s>0)
        ########### SELECT EVERY 10KM (ADDED COORD ON END)
        lat_sel = np.append(np.arange(0,len(m.latitude),10), len(m.latitude)-1)
        lon_sel = np.append(np.arange(0,len(m.longitude),10), len(m.longitude)-1)

        # select mrms 
        m = m.isel(latitude=lat_sel, longitude=lon_sel)
        m = m.drop(['step','heightAboveSea','valid_time'])
        # select rqi
        r = r.isel(latitude=lat_sel, longitude=lon_sel)
        s = s.isel(latitude=lat_sel, longitude=lon_sel)
        ########### CALC 15MIN INT
        intensity = m.unknown.resample(time='1T').asfreq().fillna(0)
        intensity = intensity.rolling(time=15,min_periods=1).sum()*(60/15)

        ########### CONVERT TO DF
        m = m.unknown.to_dataframe().rename(columns={'unknown': 'accum'})
        intensity = intensity.to_dataframe().rename(columns={'unknown': 'intensity'})
        r = r.unknown.to_dataframe().rename(columns={'unknown': 'rqi'})
        s = s.storm_id.to_dataframe()
        
        ########### RESAMPLE TO 8HR
        m = m.groupby([pd.Grouper(level='time', freq='8h'), 'latitude', 'longitude']).agg(list)

        # add intensity
        m['intensity'] = intensity.groupby([pd.Grouper(level='time', freq='8h'), 'latitude', 'longitude']).agg(list).intensity
        # add rqi
        m['rqi'] = r.groupby([pd.Grouper(level='time', freq='8h'), 'latitude', 'longitude']).agg(list).rqi
        # add storm_id
        m['storm_id'] = s.groupby([pd.Grouper(level='time', freq='8h'), 'latitude', 'longitude']).agg(list).storm_id
        print(m)
        ########### CALC TOTAL ACCUM AND DROP TOTAL ACCUM < 1MM
        m = m.reset_index()
        m['total_mrms_accum'] = [np.sum(m.accum[i]) for i in m.index]
        m = m.loc[m['total_mrms_accum']>1].reset_index(drop=True)

        ########### CALC ADDL ACCUM FEATURES
        m['max_accum_atgage'] = [np.max(m.accum[i]) for i in m.index]

        median_accum=[]
        std_accum=[]
        var_accum=[]
        mean_accum=[]

        for idx in m.index:
            accumulation = np.array(m.accum[idx])
            accum_pos = accumulation[accumulation>0]
            
            median_accum.append(np.median(accum_pos))
            std_accum.append(np.std(accum_pos))
            var_accum.append(np.var(accum_pos))
            mean_accum.append(np.mean(accum_pos))
    
        m['median_accum_point']=median_accum
        m['std_accum_point']=std_accum
        m['var_accum_point']=var_accum
        m['mean_accum_point']=mean_accum


        ########### CALC INTENSITY FEATURES
        m['max_mrms'] = [np.max(m.intensity[i]) for i in m.index]

        median_int=[]
        std_int=[]
        var_int=[]
        mean_int=[]

        for idx in m.index:
            intensity = np.array(m.intensity[idx])
            int_pos = intensity[intensity>0]
            
            median_int.append(np.median(int_pos))
            std_int.append(np.std(int_pos))
            var_int.append(np.var(int_pos))
            mean_int.append(np.mean(int_pos))
            
        m['median_int_point']=median_int
        m['std_int_point']=std_int
        m['var_int_point']=var_int
        m['mean_int_point']=mean_int

        m['duration']=[pd.DataFrame(m.intensity[i]) for i in m.index]
        m['duration']=[m.duration[i].loc[m.duration[i][0]>0].index[-1]-m.duration[i].loc[m.duration[i][0]>0].index[0] for i in m.index]
        ########### CALC RQI FEATURES
         
        rqi_mean = []
        rqi_median = []
        rqi_min = []
        rqi_max = []
        rqi_std = []
        for idx in m.index:
            rqi = m.rqi[idx]
            rqi = pd.DataFrame(rqi).dropna()
            rqi_mean.append(rqi.mean().values[0])
            rqi_median.append(rqi.median().values[0])
            rqi_min.append(rqi.min().values[0])
            rqi_max.append(rqi.max().values[0])
            rqi_std.append(rqi.std().values[0])
        m['rqi_mean'] = rqi_mean
        m['rqi_median'] = rqi_median
        m['rqi_min'] = rqi_min
        m['rqi_max'] = rqi_max
        m['rqi_std'] = rqi_std
        
        ########### CALC ADDL POINT AND STORM FEATURES IN ANOTHER FILE

        ########### CLEAN AND SAVE
        m = m.drop(columns=['accum', 'intensity', 'rqi'])

        m['storm_id'] = [pd.Series(m.storm_id[i]).dropna().unique() for i in m.index] #drop na

        # month
        m['month']=[m['time'][i].month for i in m.index]
        # hour
        m['hour']=[m['time'][i].hour for i in m.index]

        m.to_feather('state/'+str(year)+'_'+month)



# %%
