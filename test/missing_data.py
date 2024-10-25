#%%
import pandas as pd
import xarray as xr
#%%
missing = []
missing_pos = []

for year in range(2018, 2024):
    for month in ['may', 'jun', 'jul', 'aug', 'sep']:
        dataset = '../../data/MRMS/2min_rate_cat_month_CO/'+str(year)+'_'+month+'_rate_CO.grib2'

        ds = xr.open_dataset(dataset, chunks={'time': '1GB'})

        ds_min = ds.resample(time='8H').min()
        ds = ds*(2/60)
        ds_max = ds.resample(time='8H').sum()

        df1 = ds_min.where((ds_min<0) ).to_dataframe().unknown
        df2 = ds_min.where((ds_min<0) & (ds_max>1)).to_dataframe().unknown

        print(df1.dropna())
        print(df2.dropna())

        missing.append(df1.dropna())
        missing_pos.append(df2.dropna())


# %%
df=pd.concat(missing_pos)
df = df.reset_index()
df.to_feather('../output/contains_missing')
count = df.groupby(['latitude','longitude']).agg(list)
count['count'] = [len(count.iloc[i].time) for i in range(len(count))]
count['freq'] = count['count']/(459*6)
count.to_xarray()['freq'].plot()


#%%
df = pd.read_feather('../output/contains_missing')
train = pd.read_feather('../output/train_test2')

state_input = pd.read_feather('../output/state')

df_state = state_input.loc[state_input.rqi_min>=0]
df_state = df_state.dropna()
df_state = df_state.loc[(df_state.mrms_lat!=40.57499999999929)&(df_state.mrms_lon!=254.91499899999639)]

train = train.rename(columns={'start':'time','mrms_lat':'latitude','mrms_lon':'longitude'})
missing_train = pd.merge(df, train, on=['time', 'latitude', 'longitude'])

state = df_state.rename(columns={'mrms_lat':'latitude','mrms_lon':'longitude'})
missing_state = pd.merge(df, state, on=['time', 'latitude', 'longitude'])
#%%
missing_train['year'] = missing_train.time.dt.year

# how many missing values in 8-hr chunks in training

missing_win = []

for year in range(2018, 2024):
    print(year)
    for idx,month in enumerate(['may', 'jun', 'jul', 'aug', 'sep']):

        m = missing_train[(missing_train.year==year)&(missing_train.month==idx+5)]

        dataset = '../../data/MRMS/2min_rate_cat_month_CO/'+str(year)+'_'+month+'_rate_CO.grib2'

        ds = xr.open_dataset(dataset, chunks={'time': '1GB'})

        for time in m.index:
            s = m.time[time]
            lat = m.latitude[time]
            lon = m.longitude[time]
            ds_slice = ds.sel(time=slice(s,s+pd.Timedelta(hours=8)),latitude = lat,longitude=lon)
            d = pd.DataFrame(data = {'rate':ds_slice.unknown.values})
            d['year'] = year
            d['month'] = month
            missing_win.append(ds_slice.unknown.values)



# %%
import numpy as np
ab = []
for i in range(len(missing_win)):
    # Example DataFrame with missing indicator -3
    data = {'values': missing_win[i]}
    df = pd.DataFrame(data)

    # Find indices where the value is -3 (missing indicator)
    missing_indices = df[df['values'] == -3].index

    # Get values before and after each -3
    before_after = []
    for idx in missing_indices:
        # Get the previous value if it exists
        before = df['values'][idx - 1] if idx - 1 >= 0 else np.nan
        # Get the next value if it exists
        after = df['values'][idx + 1] if idx + 1 < len(df) else np.nan
        before_after.append({'before': before, 'missing': -3, 'after': after})

    # Create a DataFrame to display results
    result_df = pd.DataFrame(before_after)

    ab.append(result_df)

ab = pd.concat(ab)
ab[ab.after>=0].mean()
ab[ab.before>=0].mean()
#%%
timesteps=[]
for year in range(2018, 2024):

    for month in ['may', 'jun', 'jul', 'aug', 'sep']:
        dataset = '../../data/MRMS/2min_rate_cat_month_CO/'+str(year)+'_'+month+'_rate_CO.grib2'

        ds = xr.open_dataset(dataset, chunks={'time': '1GB'})

        ds_min = ds.resample(time='8H').min()
        print(month)
        print(year)
        print(len(ds_min.time))

        timesteps.append([month,year,len(ds_min.time)])
