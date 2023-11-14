import pandas as pd
from os import listdir
import numpy as np
from datetime import timedelta

def get_usgsnew(data_folder):
    # open
    filenames = listdir(data_folder+'\\other_usgs')
    filenames.sort(reverse=True)
    locations = pd.read_csv(data_folder+'\\USGS_new.csv',index_col=0)

    gage=[]
    for i in filenames:
        df = pd.read_csv(data_folder+'\\other_usgs\\'+i,sep='\t',engine='python',skiprows=30) # remove header
        df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2])
        df = df.rename({df.iloc[:, 2].name: 'datetime'}, axis='columns')
        df = df.rename({df.iloc[:, 4].name: 'accum'}, axis='columns')
        df = df.loc[df.accum!='Ssn'] # remove 'ssn', meaning not recorded during offseason
        df.accum = df.accum.astype('float')
        df.accum = df.accum*25.4 #convert to mm from in
        df = df.drop(df.columns[[0,1,3,5]],axis = 1)

        df = df.set_index('datetime')
        df = df[~df.index.duplicated()] # remove duplicated timesteps
        df = df.resample('1Min').asfreq().fillna(0)
        rate = df.accum.rolling(15,min_periods=1).sum()
        rate = (rate*(60/15))
        df['15_int'] = rate

        gage.append(df)

    coord = zip(locations.latitude,locations.longitude)
    usgs = dict(zip(coord, gage))
    return usgs
