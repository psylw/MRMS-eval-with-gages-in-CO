
###############################################################################
# some gages are very close, look at close gage time series to make sure they're different gages
###############################################################################

###############################################################################

import pickle
import pandas as pd
import matplotlib.pyplot as plt

#####################################################################################################################   IMPORT GAGE DATA

# Load the dictionary from the Pickle file
with open('..//output//gage_all.pickle', 'rb') as file:
    gage = pickle.load(file)

meta = []
for i in range(len(gage)):
    g = gage[i]
    if i!=2:
        for j in range(len(g)):
            meta.append([i,j,g[j][0],g[j][1],g[j][2],g[j][3]])

meta = pd.DataFrame(meta,columns=['i','j','source','year','lat','lon'])

meta = meta.round(3)

dup = []
for yr in range(2018,2024):
    test = meta.loc[meta.year==yr]
    duplicates = test[test.duplicated(subset=['lat', 'lon'], keep=False)]
    dup.append(duplicates)

dup = pd.concat(dup).sort_values(by='lat')

for i in dup.lat.unique():
    plot = dup.loc[dup.lat==i]
    
    plot1 = plot.iloc[0]
    plot2 = plot.iloc[1]

    plt.plot(gage[plot1.i][plot1.j][4].accum-gage[plot2.i][plot2.j][4].accum)
    plt.show()