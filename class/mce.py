import pandas as pd
import numpy as np
from datetime import datetime,timedelta

def mce(df):
    mce_sorted = []
    for i in df.index:
        gm = pd.DataFrame(data=[df.gage[i],df.mrms[i]]).T.rename(columns={0:'gage',1:'mrms'})
        gm = gm.loc[(gm.gage>0)|(gm.mrms>0)]

        datetime_index = pd.date_range(start='2023-11-15', periods=len(gm), freq='1T')

        gm['dt'] = datetime_index
        
        gm = gm.set_index('dt',drop=True)
        # only look at samples where positive
        
        gm_r = gm.resample('20min').max()

        g_r = gm_r.gage.values
        m_r = gm_r.mrms.values

        #g_sort = np.sort(g_r)
        #m_sort = np.sort(m_r)
        
        mce_sorted.append(1-(np.mean(np.abs(m_r - g_r))/np.mean(np.abs(g_r - np.mean(g_r)))))
    return mce_sorted