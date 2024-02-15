
import xarray as xr
import os
import glob

storm_folder = os.path.join('..', '..','..',"storm_stats")

file_storm = glob.glob(storm_folder+'//'+'*.nc')
#%% MOVE THIS
# save coordinates, DELETE other storm files #MOVE TO SAVE_STORMID.PY
# 12 min/file, takes most memory
for file in file_storm:
    storm = xr.open_dataset(file,chunks={'time': '10MB'})
    storm = storm.where(storm>0,drop=True)
    storm = storm.to_dataframe()

    storm = storm.loc[storm.storm_id>0]
    storm = storm.reset_index().groupby(['storm_id']).agg(list)
    storm = storm.reset_index()

    name = '//'+file[-28:-3]+'_coord'

    storm.to_feather(storm_folder+name)