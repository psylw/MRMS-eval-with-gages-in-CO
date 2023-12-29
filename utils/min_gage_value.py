import os
import xarray as xr
from dask.distributed import Client
import matplotlib.pyplot as plt
from os import listdir
#import rioxarray as rxr
import glob
#import seaborn as sns
#from shapely.geometry import MultiPoint
#import geopandas as gpd
#import cartopy.crs as ccrs

from CPF import *
from utils.clean_gage_data.grizzly import *
from coagmet import *
from disdrometer import *
from archive.mrms_gage import *
from usgs_other import *
from elevation import *
from usgs_new import *
from delete.ET import *

# Create a path to the code file
codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)

gage_folder = os.path.join(parentDir,"precip_gage")
#gage_folder = 'C:\\Users\\GSCOB\\OneDrive - Colostate\\Desktop\\precip_gage'

# grizzly
grizzly = get_grizzly(gage_folder)
# cpf
cpf = get_cpf(gage_folder+'\\cpf google drive')
# disdrometer
disdrom = get_disdrom(gage_folder)
# coagmet
coag = get_coagmet(gage_folder)
# other gages
other = get_usgs_other(gage_folder)

new_usgs = get_usgsnew(gage_folder)

# bring everything together
gage = {**grizzly, **cpf, **disdrom, **coag, **other, **new_usgs}
#gage = {**grizzly, **cpf, **disdrom, **coag, **other}

# get list of keys (lat/lon)
coord = [i for i in gage.keys()]

# create list of timedeltas to correct gage time error
g = ['usgs' for i in range(len(grizzly))]
c = ['csu' for i in range(len(cpf))]
d = ['csu' for i in range(len(disdrom))]
co = ['coagmet' for i in range(len(coag))]
o = ['usgs' for i in range(len(other))]
u = ['usgs' for i in range(len(new_usgs))]

s= g+c+d+co+o+u

name = 'gage_source.csv'
output = parentDir+name
pd.DataFrame(data ={'source':s}).to_csv(output)

# compare min intensity to min accum
min_accum=[]
for i in range(len(coord)):
    df = gage[coord[i]]
    df = df.loc[df['accum']>1*10**-3]
    min_accum.append(df['accum'].min())

min_int=[]
for i in range(len(coord)):
    df = gage[coord[i]]
    df = df.loc[df['15_int']>1*10**-3]
    min_int.append(df['15_int'].min())

lat_m,lon_m = nearest(m,coord)

min_accum = pd.DataFrame(data={'min_accum':min_accum,'coord':coord})

min_accum['latitude'] = lat_m
min_accum['longitude'] = lon_m

name = 'min_accum_gage'
output = parentDir+name
min_accum.to_feather(output)