import pandas as pd
from os import listdir
import os
import numpy as np
from datetime import timedelta
import glob

codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)
data_folder = os.path.join(parentDir,"precip_gage")

def get_ET(data_folder):
    locations = pd.read_csv(data_folder+'\\ET_meta.csv',index_col=0)
    filenames = listdir(data_folder+'\\East Troublesome')
    gage_id = np.unique([filenames[i][0:3] for i in range(len(filenames))])
    gage=[]
    for j in range(len(gage_id)):

        # get all files for gage and concat
        g = []
        res = [i for i in filenames if gage_id[j] in i]
        for i in res:
            g.append(pd.read_csv(data_folder+'\\East Troublesome\\'+i,skiprows=1).iloc[:,0:3])

        df = pd.concat(g)
        df.iloc[:, 1] = pd.to_datetime(df.iloc[:, 1])
        df = df.rename({df.iloc[:, 1].name: 'datetime'}, axis='columns')
        df = df.rename({df.iloc[:, 2].name: 'accum'}, axis='columns')
        df.accum = df.accum.astype('float')
        df.accum = df.accum*25.4 #convert to mm from in

        df = df.set_index('datetime')
        df = df[~df.index.duplicated()] # remove duplicated timesteps
        df = df.resample('1Min').asfreq().fillna(0)
        rate = df.accum.rolling(15,min_periods=1).sum()
        rate = (rate*(60/15))
        df['15_int'] = rate
        df = df.drop(columns='#')

        gage.append(df)
        
    coord = zip(locations.Latitude,locations.Longitude)
    ET = dict(zip(coord, gage))    
    return ET