# %%
import pandas as pd
from os import listdir
import numpy as np
from datetime import timedelta
#from mrms_gage import gage_storm
import os

data_folder = os.path.join('..', '..','..','precip_gage')
#def get_cpf(data_folder):

meta = pd.read_csv(data_folder+'\\csu_meta.csv')
meta = meta.loc[meta.Type=='rain']
filenames = listdir(os.path.join(data_folder,'csu'))
filenames.sort(reverse=True)
# %%
for i in filenames:
    print(i)
    print(pd.read_csv(os.path.join(data_folder,'csu\\')+i))

# %%
gage = []
for yr in range(2018,2024):

filenames = listdir(data_folder+'\\raw')
filenames.sort(reverse=True)
# get names
filenames = [filenames.split('.', 1)[0].lower() for filenames in filenames]

# open
gage_location = pd.read_csv(data_folder +'\\Metadata.csv')
# get rain
gage_location = gage_location.loc[gage_location.Type=='rain']
# drop columns I don't care about
gage_location = gage_location.drop(gage_location.columns[4:11], axis=1).drop(['Type'],axis=1)
# remove _ from two letter locations
gage_location.Name = [name.replace("_","") for name in gage_location.Name]
# get locations for precip gages
gage_location = gage_location[gage_location['Name'].isin(filenames)]
# remove duplicates
gage_location = gage_location.drop_duplicates(keep='first').reset_index().drop(['index'],axis=1)
# convert to universal coords
gage_location['Longitude'] = gage_location['Longitude'].to_numpy()+360
gage_location.Latitude = gage_location.Latitude.astype(type(gage_location.Longitude[0]))

# open gage data and save as list to variable
gage_raw = [pd.read_csv(data_folder +'\\raw\\'+ gage_location.Name[i]+'.csv') for i in gage_location.index]

# change datetime type
for i in gage_location.index:
    gage_raw[i].datetime = gage_raw[i].datetime.astype('datetime64[ns]')

gage_raw = [gage_raw[i].set_index('datetime') for i in gage_location.index]

# duplicate should be added, assuming it means more than one tip in minute
gage_raw = [gage_raw[i].sort_index().groupby(gage_raw[i].index).sum() for i in gage_location.index]
gage_raw = [gage_raw[i].resample('1Min').asfreq().fillna(0) for i in range(len(gage_raw))]
#gage_raw = [gage_raw[i].resample('1Min').asfreq() for i in range(len(gage_raw))]

for i in range(len(gage_raw)):
    # multiply tips by 0.3 mm
    accum = gage_raw[i]*0.3
    # calculate desired intensity by summing 1min accum with rolling desired intensity window,
    # then divide by desired intensity (min) and convert to mm/hr
    rate = accum.rolling(15,min_periods=1).sum()
    rate = (rate*(60/15))

    gage_raw[i]['15_int'] = rate
    gage_raw[i]['accum'] = accum


cpf = [gage_raw[i].drop(['tips'],axis=1) for i in range(len(gage_raw))]

cpf = [cpf[i].iloc[(cpf[i].index.year<=2022) & (cpf[i].index.year>=2021)&
                (cpf[i].index.month>=4) & (cpf[i].index.month<11)] for i in range(len(gage_raw))]

cpf.index = cpf.index+np.timedelta64(1, 'h') # shift hour bc time in MST

coord = zip(gage_location.Latitude,gage_location.Longitude)
cpf = dict(zip(coord, cpf))

    #return cpf
