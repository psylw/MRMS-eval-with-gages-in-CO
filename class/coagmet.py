import pandas as pd
import numpy as np
import os

#data_folder = os.path.join('..', '..','..','precip_gage')

def get_coagmet(data_folder):
    meta = pd.read_csv(data_folder+'\\meta_coagmet')

    gage = []
    for yr in range(2018,2024):
        year = pd.read_csv(data_folder+'\\coagmet\\'+str(yr)+'_coagmet',header=1)
        
        year[year.mm < 0] = 0 # set missing to zero

        ids = [x for x in year.id.unique() if x != 0] # zero snuck into ids, header ?

        for j in ids:
            year_gage = year.loc[year.id==j].reset_index().rename(columns={"mm": "accum",'date time':'datetime'})

            lat = meta.loc[meta.Station==j]['Latitude (degN)'].values[0]
            lon = meta.loc[meta.Station==j]['Longitude (degE)'].values[0]+360

            try:
                # select times after first positive and before last negative for year, a check that gage actually recording
                year_gage = year_gage[year_gage.loc[year_gage.accum>0].index[0]:year_gage.loc[year_gage.accum>0].index[-1]]

                year_gage = year_gage.drop(columns=['index','id'])

                if len(year_gage)>0 and lon<255.5:
                    year_gage['datetime'] = pd.to_datetime(year_gage['datetime'])
                    year_gage = year_gage.set_index('datetime')
                    year_gage = year_gage.iloc[year_gage.index.month.isin(range(5,10))]

                    year_gage = year_gage.resample('1Min').asfreq().fillna(0)
                    year_gage['15_int'] = (year_gage.accum.rolling(15,min_periods=1).sum())*(60/15)

                    gage.append(['coagmet',yr, lat,lon,year_gage])
            except:
                pass
    return gage



