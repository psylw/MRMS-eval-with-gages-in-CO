import pandas as pd
import numpy as np

def nrmsd(df):
    rmse_unsorted = []

    def rmse(x1,x2,gm):
        rmse = np.sqrt(np.sum((x1-x2)**2)/len(gm))
        return rmse

    for i in df.index:
        print(i/len(df))
        #id = df.gage_id[i][0]
        #max_value = max_values.loc[max_values.gage_id == gid].max_value.iloc[0]
        
        gm = pd.DataFrame(data=[df['15_int'][i],df['mrms_15_int'][i]]).T.rename(columns={0:'gage',1:'mrms'})
        #max_value = np.max(gm.mean())

        datetime_index = pd.date_range(start='2023-11-15', periods=len(gm), freq='1T')

        gm['dt'] = datetime_index
        
        gm = gm.set_index('dt',drop=True)
        
        # resample to 10min to decrease temporal sampling error
        gm = gm.resample('10min').max()
        # only look at samples where positive
        gm = gm.loc[(gm.gage>0)|(gm.mrms>0)]

        g = gm.gage.values
        m = gm.mrms.values

        #rmse_unsorted.append(rmse(g_r,m_r,gm)/max_value)
        rmse_unsorted.append(rmse(g,m,gm))
    return rmse_unsorted