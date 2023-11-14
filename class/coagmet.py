import pandas as pd
import numpy as np
#from mrms_gage import gage_storm

def get_coagmet(data_folder):
    gage_raw = pd.read_feather(data_folder+'\\coagmet_5min_01apr21_01oct22')
    gage_location = pd.read_feather(data_folder+'\\coagmet_5min_metadata')

    # remove mtr01, only recording 0.25
    gage_location = gage_location.loc[gage_location.Station!='mtr01']

    gage_location = gage_location.loc[gage_location['Longitude (degE)']<=-104]
    gage_location = gage_location.loc[gage_location.Station.isin(gage_raw.Station)]

    gage_raw = [gage_raw.loc[gage_raw.Station==i] for i in gage_location.Station]

    # get rid of -999 values
    for i in range(len(gage_raw)):
        gage_raw[i].loc[gage_raw[i].Precip<0] = 0
    
    # remove gages not recording anything
    gage_sum = [gage_raw[i].Precip.sum() for i in range(len(gage_raw))]
    gage_location['sum']=gage_sum
    gage_location=gage_location.loc[gage_location['sum']!=0]

    gage_raw = pd.concat(gage_raw)
    gage_raw = [gage_raw.loc[gage_raw.Station==i] for i in gage_location.Station]

    gage_raw = [gage_raw[i].set_index('datetime') for i in range(len(gage_raw))]
    gage_raw = [gage_raw[i].resample('1Min').asfreq().fillna(0) for i in range(len(gage_raw))]
    #gage_raw = [gage_raw[i].resample('1Min').asfreq() for i in range(len(gage_raw))]

    gage_raw = [gage_raw[i].drop(['Station'],axis=1) for i in range(len(gage_raw))]

    gage_raw = [gage_raw[i].rename(columns={"Precip": "accum"}) for i in range(len(gage_raw))]

    for i in range(len(gage_raw)):
        # calculate desired intensity by summing 1min accum with rolling desired intensity window, 
        # then divide by desired intensity (min) and convert to mm/hr
        rate = gage_raw[i].accum.rolling(15,min_periods=1).sum()
        rate = (rate*(60/15))
        gage_raw[i]['15_int'] = rate

    lat = gage_location['Latitude (degN)']
    lon = gage_location['Longitude (degE)']+360

    gage_raw = [gage_raw[i].iloc[(gage_raw[i].index.year<=2022) & (gage_raw[i].index.year>=2021)&
                (gage_raw[i].index.month>=4) & (gage_raw[i].index.month<11)] for i in range(len(gage_raw))]

    gage_raw.index = gage_raw.index+np.timedelta64(1, 'h') # shift hour bc time in MST

    coord = zip(lat,lon)

    coagmet = dict(zip(coord, gage_raw))
    return coagmet

