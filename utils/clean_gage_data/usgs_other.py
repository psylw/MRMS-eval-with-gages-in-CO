
import pandas as pd
from os import listdir
import numpy as np
from datetime import timedelta
import os

#data_folder = os.path.join('..', '..','..','precip_gage')

def get_usgs_other(data_folder):
    filenames = listdir(data_folder+'\\usgs_fromFR')

    filenames_m = listdir(data_folder+'\\usgs_fromFR_meta')
    meta = [pd.read_csv(data_folder +'\\usgs_fromFR_meta\\'+i).sort_values(by=['station_id']) for i in filenames_m]
    meta = pd.concat(meta)
    gage = []

    for i in range(len(filenames)):
        g = pd.read_csv(data_folder +'\\usgs_fromFR\\'+filenames[i])

        gage_id = filenames[i].split('_')[1]

        lat = meta.loc[meta.station_id==gage_id].lat.values[0]
        lon = meta.loc[meta.station_id==gage_id].lon.values[0]+360

        g = g.drop(g.columns.difference(['time','intensity_15m','precip_mm_diff']), 1)

        g = g.rename(columns={"time": "datetime","intensity_15m": "15_int","precip_mm_diff": "accum"})

        g['datetime'] = pd.to_datetime(g['datetime']) + pd.Timedelta(hours=6)

        g = g.set_index('datetime')

        for yr in range(2018,2024):
            year_gage = g.iloc[g.index.year==yr]
            year_gage = year_gage.iloc[year_gage.index.month.isin(range(5,10))]

            try:
                year_gage = year_gage[year_gage.loc[year_gage.accum>0].index[0]:year_gage.loc[year_gage.accum>0].index[-1]]    

                if len(year_gage)>0 and lon<255.5:
                    gage.append(['usgs',yr, lat,lon,year_gage])
            except:
                pass


    return gage

