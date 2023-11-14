
import os
from os import listdir
import sys
import pickle
sys.path.append('class')

from CPF import *
from grizzly import *
from coagmet import *
from disdrometer import *
from mrms_gage import *
from usgs_other import *
from usgs_new import *

# Create a path to the code file
codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)

gage_folder = os.path.join(parentDir,"precip_gage")

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


# Saving the dictionary to a file using Pickle
with open('output//gage_all.pickle', 'wb') as file:
    pickle.dump(gage, file)