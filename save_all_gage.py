# %%
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
from grizzly import *
from coagmet import *
from disdrometer import *
from mrms_gage import *
from usgs_other import *
from elevation import *
from usgs_new import *
from ET import *

# Create a path to the code file
codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)

gage_folder = os.path.join(parentDir,"precip_gage")
#gage_folder = 'C:\\Users\\GSCOB\\OneDrive - Colostate\\Desktop\\precip_gage'
# %%
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