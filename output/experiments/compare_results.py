#%%
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
test_files = glob.glob('test_*')
train_files = glob.glob('train_*')

# %%
t = []
t_norm = []
t_me = []
for file in train_files:
    if 'nr_' in file:
        t_norm.append(pd.read_feather(file).drop([3,4,5,7]).set_index('name')['mean_ql'].rename(file[6::]))
    elif 'mean_error' in file:
        t_me.append(pd.read_feather(file).drop([3,4,5,7]).set_index('name')['mean_ql'].rename(file[6::]))        
    else:
        t.append(pd.read_feather(file).drop([3,4,5,7]).set_index('name')['mean_ql'].rename(file[6::]))

train = pd.concat(t,axis=1).T.rename(columns={'gbt':'FO'}).sort_values(by='FO')
train_norm = pd.concat(t_norm,axis=1).T.rename(columns={'gbt':'FO'}).sort_values(by='FO')
train_me = pd.concat(t_me,axis=1).T.rename(columns={'gbt':'FO'}).sort_values(by='FO')

t = []
t_norm = []
t_me = []
for file in test_files:
    if 'nr_' in file:
        t_norm.append(pd.read_feather(file).drop([3,4,5,7]).set_index('name')['ql'].rename(file[5::]))
    elif 'mean_error' in file:
        t_me.append(pd.read_feather(file).drop([3,4,5,7]).set_index('name')['ql'].rename(file[5::]))        
    else:
        t.append(pd.read_feather(file).drop([3,4,5,7]).set_index('name')['ql'].rename(file[5::]))

test = pd.concat(t,axis=1).T.rename(columns={'gbt':'FO'}).sort_values(by='FO')
test_norm = pd.concat(t_norm,axis=1).T.rename(columns={'gbt':'FO'}).sort_values(by='FO')
test_me = pd.concat(t_me,axis=1).T.rename(columns={'gbt':'FO'}).sort_values(by='FO')

# %%
