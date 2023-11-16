
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime,timedelta

codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
parentDir = os.path.dirname(codeDir)

compare = pd.read_feather('G:\PSW\working code\MRMS-eval-with-gages-in-CO\output\window_values_FN')

# see what mce distribution looks like when MRMS FN
df = compare.loc[(compare.total_accum_atgage>1)|(compare.total_gage_accum>1)]

# %%
current_datetime = datetime.now()
mce_raw = []
mce_sorted = []
mce_unsorted = []
for i in df.index:
    gm = pd.DataFrame(data=[df.gage[i],df.mrms[i]]).T.rename(columns={0:'gage',1:'mrms'})
    gm = gm.loc[(gm.gage>0)|(gm.mrms>0)]

    datetime_index = pd.date_range(start='2023-11-15', periods=len(gm), freq='1T')

    gm['dt'] = datetime_index
    
    gm = gm.set_index('dt',drop=True)
    # only look at samples where positive
    
    gm_r = gm.resample('10min').max()
    
    g = gm.gage.values
    m = gm.mrms.values

    g_r = gm_r.gage.values
    m_r = gm_r.mrms.values

    g_sort = np.sort(g_r)
    m_sort = np.sort(m_r)
    
    mce_raw.append(1-(np.mean(np.abs(m - g))/np.mean(np.abs(g - np.mean(g)))))
    mce_unsorted.append(1-(np.mean(np.abs(m_r - g_r))/np.mean(np.abs(g_r - np.mean(g_r)))))
    mce_sorted.append(1-(np.mean(np.abs(m_sort - g_sort))/np.mean(np.abs(g_sort - np.mean(g_sort)))))

# %%
#df['norm_diff'] = mce_raw
#print('raw')
df['norm_diff'] = mce_sorted
print('sorted')
#df['norm_diff'] = mce_unsorted
#print('unsorted')
df['total_gage_accum'] = [np.sum(df.gage_accum[i]) for i in df.index]

test = df.loc[(df.norm_diff>.4)]

print(len(test))
# how often is mce high just because gage variability high?
print(len(test.loc[(test.total_accum_atgage<1)&(test.total_gage_accum>5)]))
# how often is mce low when values are close?
test2 = df.loc[(df.total_accum_atgage-df.total_gage_accum<.5)&(df.norm_diff<.4)&(df.max_mrms>5)]
print(len(test2))
'''for i in range(10): # plot random sample of 10 timeseries
    plot = test2.sample(1)
    plt.plot(plot.iloc[0].mrms)
    plt.plot(plot.iloc[0].gage)
    plt.title('total accum diff low, low mce')
    plt.show()

for i in range(10): # plot random sample of 10 timeseries
    plot = test.loc[test.total_accum_atgage>10].sample(1)
    plt.plot(plot.iloc[0].mrms)
    plt.plot(plot.iloc[0].gage)
    plt.title('example good above 10 mm')
    plt.show()'''

# %%
df['mce_unsorted'] = mce_unsorted
df['mce_raw'] = mce_raw
df['mce_sorted'] = mce_sorted

df.loc[df.mce_unsorted<0,['mce_unsorted']]=-.1
df.loc[df.mce_raw<0,['mce_raw']]=-.1
df.loc[df.mce_sorted<0,['mce_sorted']]=-.1

'''hx = np.histogram(df.mce_sorted,bins=np.arange(-.1,1.1,.1))
hu = np.histogram(df.mce_unsorted,bins=np.arange(-.1,1.1,.1))
hr = np.histogram(df.mce_raw,bins=np.arange(-.1,1.1,.1))'''

hx = np.histogram(df.mce_sorted,bins=np.arange(0,1.1,.1))
hu = np.histogram(df.mce_unsorted,bins=np.arange(0,1.1,.1))
hr = np.histogram(df.mce_raw,bins=np.arange(0,1.1,.1))

import matplotlib as mpl

# set figure defaults
mpl.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12.0/2, 10.0/2)

fig, ax = plt.subplots(figsize=(6, 6))


plt.xlabel('MCE')
plt.ylabel('frequency')

plt.bar(hx[1][:-1],hx[0].astype(float)/len(df),edgecolor = 'k', color = [], width = .1, linewidth = 2,label='sorted,downsampled')
plt.bar(hu[1][:-1],hu[0].astype(float)/len(df),edgecolor = 'r', color = [], width = .1, linewidth = 2,label='unsorted,downsampled')
plt.bar(hr[1][:-1],hr[0].astype(float)/len(df),edgecolor = 'b', color = [], width = .1, linewidth = 2,label='raw')
plt.legend()

# %%
