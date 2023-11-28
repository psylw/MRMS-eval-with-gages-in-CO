
from os import listdir
import numpy as np
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
#from mrms_gage import gage_storm
#data_folder = os.path.join('..', '..','..','precip_gage')
def get_grizzly(data_folder):
    filenames = listdir(data_folder +'\\usgs_grizzly')

    gage_id = [filenames[i][0:5] for i in range(len(filenames))]
    # read attribute table
    meta = pd.read_csv(data_folder +'\\usgs_grizzly_meta.csv')

    gage = []
    for id in gage_id:
        g = pd.read_csv(data_folder +'\\usgs_grizzly\\'+id+'_20210727_20210930_checked_processed.csv')

        lat = meta.loc[meta.ID==id].Lat.values[0]
        lon = meta.loc[meta.ID==id].Long.values[0]+360

        g = g.drop(g.columns.difference(['TimeStamp (Local)','15-minute Intensity (mm/h)','Bin Accum (mm)']), 1)

        g = g.rename(columns={"TimeStamp (Local)": "datetime","15-minute Intensity (mm/h)": "15_int","Bin Accum (mm)": "accum"})
        
        g['datetime'] = pd.to_datetime(g['datetime']) + pd.Timedelta(hours=6)
        g = g.set_index('datetime')

        g = g.iloc[g.index.month.isin(range(5,10))]

        g = g[g.loc[g.accum>0].index[0]:g.loc[g.accum>0].index[-1]]

        gage.append(['usgs',2021, lat,lon,g])
    return gage

