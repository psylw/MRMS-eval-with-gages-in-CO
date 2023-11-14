import pandas as pd
from os import listdir
import numpy as np
from datetime import timedelta

def get_usgs_other(data_folder):
    filenames = listdir(data_folder+'\\other_usgs_fromFrancis')
    filenames.sort(reverse=True)
    gage = [pd.read_csv(data_folder +'\\other_usgs_fromFrancis\\'+i) for i in filenames]
    gage = [gage[i].drop(gage[i].columns.difference(['time','precip_mm_diff','intensity_15m']), 1) for i in range(len(gage))]

    gage = [gage[i].rename(columns={"time": "datetime","intensity_15m": "15_int","precip_mm_diff": "accum"}
                    ) for i in range(len(gage))]

    for i in range(len(gage)):
        gage[i].datetime = gage[i].datetime.astype('datetime64[ns]')

    gage = [gage[i].set_index('datetime') for i in range(len(gage))]

    station = [filenames[i][8:17] for i in range(len(filenames))]
    station = [station[i].split('_')[0] for i in range(len(filenames))]

    filenames = listdir(data_folder+'\\meta_fromFrancis')
    filenames.sort(reverse=True)

    meta = [pd.read_csv(data_folder +'\\meta_fromFrancis\\'+i).sort_values(by=['station_id']) for i in filenames]

    meta = pd.concat(meta)

    gage_location = meta.loc[meta.station_id.isin(station)].drop('last_data',axis=1)

    gage_location['lon'] = gage_location['lon']+360

    coord = zip(gage_location.lat,gage_location.lon)
    gage = dict(zip(coord, gage))


    return gage
