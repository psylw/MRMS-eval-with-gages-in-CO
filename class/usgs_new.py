import pandas as pd
from os import listdir
import os
import numpy as np
from datetime import timedelta
#data_folder = os.path.join('..', '..','..','precip_gage')
def get_usgsnew(data_folder):
    # open
    filenames = listdir(data_folder+'\\usgs_public')
    filenames.sort(reverse=True)
    meta = pd.read_csv(data_folder+'\\usgs_public_meta.csv',index_col=0).reset_index()

    gage=[]
    for i in filenames:
        lat = meta.loc[meta['Site Number']==float(i)].lat.values[0]
        lon = meta.loc[meta['Site Number']==float(i)].lon.values[0]+360

        df = pd.read_csv(data_folder+'\\usgs_public\\'+i,sep='\t',engine='python',skiprows=30) # remove header
        df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2]) + pd.Timedelta(hours=6) # MDT to UTC

        df = df.rename({df.iloc[:, 2].name: 'datetime'}, axis='columns')
        df = df.rename({df.iloc[:, 4].name: 'accum'}, axis='columns')
        df = df.loc[df.accum!='Ssn'] # remove 'ssn', meaning not recorded during offseason
        df = df.loc[df.accum!='Eqp']
        df.accum = df.accum.astype('float')
        df.accum = df.accum*25.4 #convert to mm from in
        df = df.drop(df.columns[[0,1,3,5]],axis = 1)

        df = df.set_index('datetime')
        df = df[~df.index.duplicated()] # remove duplicate timesteps

        for yr in range(2018,2024):
            year_gage = df.iloc[df.index.year==yr]
            year_gage = year_gage.iloc[year_gage.index.month.isin(range(5,10))]
            try:
                year_gage = year_gage[year_gage.loc[year_gage.accum>0].index[0]:year_gage.loc[year_gage.accum>0].index[-1]]

                if len(year_gage)>0 and lon<255.5:
                    year_gage = year_gage.resample('1Min').asfreq().fillna(0)
                    
                    year_gage['15_int'] = (year_gage.accum.rolling(15,min_periods=1).sum())*(60/15)

                    gage.append(['usgs',yr, lat,lon,year_gage])
            except:
                pass
    return gage