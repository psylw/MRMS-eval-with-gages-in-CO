from os import listdir
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd
#from mrms_gage import gage_storm

def get_grizzly(data_folder):
    filenames = listdir(data_folder +'\\gages')
    filenames.sort(reverse=True)
    gage_id = [filenames[i][0:5] for i in range(len(filenames))]
    # read attribute table
    gage_location = pd.read_csv(data_folder +'\\Grizzly_attributes.csv')
    # drop extra data
    gage_location = gage_location.drop(columns=['Gauge_Name', 'Gauge_Name','Fire','Source',
    'Operator','Elevation','Label_Name'])
    # select gages of interest
    gage_location = gage_location[gage_location['ID'].isin(gage_id)]
    # replace longitude in datafram with global crs
    gage_location['Long'] = gage_location['Long'].to_numpy()+360
    gage_id = gage_location['ID'].values

    gage = [pd.read_csv(data_folder +'\\gages\\'+i+'_20210727_20210930_checked_processed.csv') for i in gage_id]

    gage = [gage[i].drop(gage[i].columns.difference(['TimeStamp (Local)','15-minute Intensity (mm/h)','Bin Accum (mm)']), 1) for i in range(len(gage_id))]

    gage = [gage[i].rename(columns={"TimeStamp (Local)": "datetime","15-minute Intensity (mm/h)": "15_int","Bin Accum (mm)": "accum"}
                    ) for i in range(len(gage_id))]

    # change datetime type
    for i in range(len(gage_id)):
        gage[i].datetime = gage[i].datetime.astype('datetime64[ns]')

    gage = [gage[i].set_index('datetime') for i in range(len(gage_id))]

    gage = [gage[i].fillna(0) for i in range(len(gage_id))]

    coord = zip(gage_location.Lat,gage_location.Long)
    gage = dict(zip(coord, gage))

    return gage
