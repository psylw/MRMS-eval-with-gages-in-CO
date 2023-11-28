# %%
import pandas as pd
import os
import urllib.request
from urllib.request import HTTPError

gage_folder = os.path.join('..', '..','..','precip_gage')

gage_id = pd.read_csv(gage_folder+'\\usgs_public_meta.csv',index_col=0).reset_index()['Site Number'].astype('str')

gage_id = [gage_id[i].split('.')[0] for i in gage_id.index]
# %%
for i in range(len(gage_id)):
    if len(gage_id[i])<8:
        gage_id[i]=str(0)+gage_id[i]

destination = os.path.join('..', '..','..','precip_gage','usgs_public')

for filename in gage_id:
    
    url = 'https://nwis.waterservices.usgs.gov/nwis/iv/?sites='+filename+'&parameterCd=00045&startDT=2018-04-21T08:53:17.158-07:00&endDT=2023-10-21T08:53:17.158-07:00&siteStatus=all&format=rdb'


    fetched_request = urllib.request.urlopen(url)
    
    with open(destination + os.sep + filename, 'wb') as f:
        f.write(fetched_request.read())


# %%
