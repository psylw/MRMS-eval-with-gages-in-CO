# %%
import pandas as pd
from os import listdir
import numpy as np
from datetime import timedelta
#from mrms_gage import gage_storm
import os

#data_folder = os.path.join('..', '..','..','precip_gage')
def get_cpf(data_folder):
    meta = pd.read_csv(data_folder+'\\csu_meta.csv')
    meta = meta.loc[meta.Type=='rain']
    filenames = listdir(os.path.join(data_folder,'csu'))

    gage = []
    for i in filenames:
        g = pd.read_csv(os.path.join(data_folder,'csu\\')+i)
        # convert to UTC from MST
        g['datetime'] = pd.to_datetime(g['datetime']) + pd.Timedelta(hours=7)
        g = g.set_index('datetime')

        for yr in range(2018,2024):
            year_gage = g.iloc[g.index.year==yr]
            try:
                id = year_gage.site_id.unique()[0].strip("'")

                lat = float(meta.loc[meta.Name==id]['Latitude'].values[0])
                lon = meta.loc[meta.Name==id]['Longitude'].values[0]+360

                # duplicate should be added, assuming it means more than one tip in minute
                year_gage = year_gage.sort_index().groupby(year_gage.index).sum() 
                
                # multiply tips by 0.3 mm
                year_gage['accum'] = year_gage['tip']*0.3

                # select times after first positive and before last negative for year, a check that gage actually recording
                year_gage = year_gage[year_gage.loc[year_gage.accum>0].index[0]:year_gage.loc[year_gage.accum>0].index[-1]]

                year_gage = year_gage.drop(columns=['tip'])
                year_gage = year_gage.iloc[year_gage.index.month.isin(range(5,10))]

                if len(year_gage)>0 and lon<255.5:
                    year_gage = year_gage.resample('1Min').asfreq().fillna(0)

                    year_gage['15_int'] = (year_gage.accum.rolling(15,min_periods=1).sum())*(60/15)
                    gage.append(['csu',yr, lat,lon,year_gage])
            except:
                    pass
    return gage


# %%
