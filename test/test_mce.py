#import pandas as pd


# open window values
test = pd.read_feather('window_values_FN')

compare = compare.loc[(compare.total_accum_atgage>1)|(compare.total_gage_accum>1)]


# add downsampled, unsorted
# compute MSE from chunks
current_datetime = datetime.now()

mce_unsorted = []
mce_raw = []
mce_sorted = []

for i in test.index:
    g = test.gage[i]
    m = test.mrms[i]    

    datetime_g = [current_datetime + timedelta(minutes=i) for i in range(len(g))]
    datetime_m = [current_datetime + timedelta(minutes=i) for i in range(len(m))]

    # segment into 10 min chunks
    g = pd.DataFrame(data=g,index=datetime_g)
    m = pd.DataFrame(data=m,index=datetime_m)
    g_r = g.resample('10min').max().values
    m_r = m.resample('10min').max().values

    g_sort = np.sort(g_r)
    m_sort = np.sort(m_r)

    mce_raw.append(1-(np.mean(np.abs(m - g))/np.mean(np.abs(g - np.mean(g)))))
    mce_unsorted.append(1-(np.mean(np.abs(m_r - g_r))/np.mean(np.abs(g_r - np.mean(g_r)))))

    mce_sorted.append(1-(np.mean(np.abs(m_sort - g_sort))/np.mean(np.abs(g_sort - np.mean(g_sort)))))

test['mce_unsorted'] = mce_unsorted
test['mce_raw'] = mce_raw
test['sorted'] = mce_sorted

hx = np.histogram(compare.mce_sorted,bins=np.arange(0,1.1,.1))
hu = np.histogram(compare.mce_unsorted,bins=np.arange(0,1.1,.1))
hr = np.histogram(compare.mce_raw,bins=np.arange(0,1.1,.1))

import matplotlib as mpl

# set figure defaults
mpl.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12.0/2, 10.0/2)

fig, ax = plt.subplots(figsize=(6, 6))

plt.xlabel('MCE')
plt.ylabel('frequency')

plt.bar(hx[1][:-1],hx[0].astype(float)/len(test),edgecolor = 'b', color = [], width = .1, linewidth = 2,label='sorted,downsampled')
plt.bar(hu[1][:-1],hu[0].astype(float)/len(test),edgecolor = 'r', color = [], width = .1, linewidth = 2,label='unsorted,downsampled')
#plt.bar(hr[1][:-1],hr[0].astype(float)/len(test),edgecolor = 'b', color = [], width = .1, linewidth = 2,label='raw')
plt.legend()

fig.savefig("histmce.pdf",
       bbox_inches='tight',dpi=255,transparent=False,facecolor='white')