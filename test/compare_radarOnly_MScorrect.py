# %%
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

sys.path.append('../class')

from mce import mce

# open both window values
file_path = os.path.join('..', 'output', 'window_values_FN')
test = pd.read_feather(file_path)

file_path = os.path.join('..', 'output', 'window_values_radaronly')
compare = pd.read_feather(file_path)

accumulation_threshold = 1

compare = compare.loc[(compare.total_accum_atgage>accumulation_threshold)|(compare.total_gage_accum>accumulation_threshold)]

test = test.loc[(test.total_accum_atgage>accumulation_threshold)|(test.total_gage_accum>accumulation_threshold)]
# %%
# calculate mce
compare_mce = mce(compare)
test_mce = mce(test)
# %%
######################   COMPARE VALUES    ###############################################################################################

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
for i in range(100):
    i = random.randint(0, len(compare))
    
    c = compare.iloc[i]
    
    axs[0].scatter(c.gage,c.mrms)
    axs[0].set_title('radar only')
    
    t = test.loc[(test['index']==c['index'])&(test['gage_id']==c['gage_id'][0])]
    
    try:
        axs[1].scatter(t.gage.iloc[0],t.mrms.iloc[0])
        axs[1].set_title('with correction')
    except:
        pass
# %%
######################   COMPARE TIME SERIES    ###############################################################################################
for i in range(100):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    c = compare.iloc[i]
    
    axs[0].plot(c.mrms,label='mrms')
    axs[0].plot(c.gage,label='gage')
    axs[0].legend()
    axs[0].set_title(str(c.mce)[0:4])
    
    t = test.loc[(test['index']==c['index'])&(test['gage_id']==c['gage_id'][0])]
    try:
        axs[1].plot(t.mrms.iloc[0],label='mrms')
        axs[1].plot(t.gage.iloc[0],label='gage')
        axs[1].legend()
        axs[1].set_title(str(t.mce.iloc[0])[0:4])
    except:
        pass
    plt.show()

# %%
accumulation_threshold = 1

compare = compare.loc[(compare.total_accum_atgage>accumulation_threshold)|(compare.total_gage_accum>accumulation_threshold)]

test = test.loc[(test.total_accum_atgage>accumulation_threshold)|(test.total_gage_accum>accumulation_threshold)]
print(len(test))
test.loc[test.mce<0,['mce']]=-.1
compare.loc[compare.mce<0,['mce']]=-.1

h_ms = np.histogram(test.mce,bins=np.arange(0,1.1,.1))
h_r = np.histogram(compare.mce,bins=np.arange(0,1.1,.1))


import matplotlib as mpl

# set figure defaults
mpl.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12.0/2, 10.0/2)

fig, ax = plt.subplots(figsize=(6, 6))


plt.xlabel('MCE')
plt.ylabel('frequency')

plt.bar(h_ms[1][:-1],h_ms[0].astype(float)/len(test),edgecolor = 'blue', color = [], width = .1, linewidth = 2,label='with correction')
plt.bar(h_r[1][:-1],h_r[0].astype(float)/len(compare),edgecolor = 'r', color = [], width = .1, linewidth = 2,label='radar only')

plt.legend()

# %%