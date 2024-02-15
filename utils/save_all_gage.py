# %%
import os
from os import listdir
import sys
import pickle
sys.path.append('class')

from CPF import *
from grizzly import *
from coagmet import *
from disdrometer import *
from usgs_other import *
from usgs_new import *

# Create a path to the code file
data_folder = os.path.join('..', '..','precip_gage')


# coagmet
coag = get_coagmet(data_folder)
# cpf
cpf = get_cpf(data_folder)
# disdrometer
disdrom = get_disdrom(data_folder)
# grizzly
grizzly = get_grizzly(data_folder)
# from FR
other = get_usgs_other(data_folder)
# public
new_usgs = get_usgsnew(data_folder)

# bring everything together
#gage = {**coag, **cpf, **disdrom,**grizzly, **other, **new_usgs}
gage = [coag, cpf, disdrom,grizzly, other, new_usgs]

# Saving the dictionary to a file using Pickle
with open('output//gage_all.pickle', 'wb') as file:
    pickle.dump(gage, file)
# %%
