# %%
###############################################################################
# put all statewide features together
###############################################################################
import pandas as pd
import os
import rioxarray as rxr
import glob
# get all state files
file_state = os.listdir('state/')
#%%
elev = '\\CO_SRTM1arcsec__merge.tif'
data_folder = os.path.join('..','..','elev_data')

coelev = rxr.open_rasterio(data_folder+elev)
# change lon to match global lat/lon in grib file
coelev = coelev.assign_coords(x=(((coelev.x + 360))))

coelev = coelev.rename({'x':'longitude','y':'latitude'})

s = '\\CO_SRTM1arcsec_slope.tif'

coslope = rxr.open_rasterio(data_folder+s)

# change lon to match global lat/lon in grib file
coslope = coslope.assign_coords(x=(((coslope.x + 360))))

coslope = coslope.rename({'x':'longitude','y':'latitude'})

# aspect at gage
asp = '\\CO_SRTM1arcsec_aspect.tif'

coasp = rxr.open_rasterio(data_folder+asp)
# change lon to match global lat/lon in grib file
coasp = coasp.assign_coords(x=(((coasp.x + 360))))

coasp = coasp.rename({'x':'longitude','y':'latitude'})

storm_folder = os.path.join('..', '..','..',"storm_stats")
# ACCUMULATION
file_temp = glob.glob(storm_folder+'//'+'*temp_var_accum')
file_spatial = glob.glob(storm_folder+'//'+'*spatial_var_accum')

# VELOCITY
file_vel = glob.glob(storm_folder+'//'+'*velocity')
# AREA
file_area = glob.glob(storm_folder+'//'+'*area')

def find_files_by_year_and_month(year, month, filenames):
    # Convert numeric month to string representation
    month_str = {
        1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
        7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
    }[month]

    # Form the year and month strings
    year_str = str(year)

    # Filter files by matching year and month
    matching_files = [
        filename for filename in filenames
        if year_str in filename and month_str in filename
    ]

    return matching_files
# %%
########### CALC ADDL POINT AND STORM FEATURES
updated_file = []

for file in file_state:
    print(file)
    # sort columns by labeled dataset
    state = pd.read_feather('state/'+file)
    state = state.rename(columns={'latitude':'mrms_lat', 'longitude':'mrms_lon'})
    ########### CALC ADDL POINT FEATURES
    elevation = [coelev.sel(longitude=state.mrms_lon[i],latitude=state.mrms_lat[i],
                        method='nearest').values[0] for i in range(len(state))]
    state['point_elev']=elevation

    slope = [coslope.sel(longitude=state.mrms_lon[i],latitude=state.mrms_lat[i],
                        method='nearest').values[0] for i in range(len(state))]

    state['point_slope']=slope

    state.point_slope=state.point_slope.where(state.point_slope.between(0,100))

    aspect = [coasp.sel(longitude=state.mrms_lon[i],latitude=state.mrms_lat[i],
                        method='nearest').values[0] for i in range(len(state))]
    state['point_aspect']=aspect
    ########### CALC ADDL STORM FEATURES
    temp_atgage = []
    spatial_atgage = []
    velocity = []
    area = []

    for idx in state.index:
        #print(idx/len(state))
        sample = state.iloc[idx]
        month = sample.month
        year = sample.time.year
        storm_ids = sample.storm_id

        temp = pd.read_feather(find_files_by_year_and_month(year, month, file_temp)[0])
        spatial = pd.read_feather(find_files_by_year_and_month(year, month, file_spatial)[0])
        vel = pd.read_feather(find_files_by_year_and_month(year, month, file_vel)[0])    
        a = pd.read_feather(find_files_by_year_and_month(year, month, file_area)[0])

        temp_atgage.append(temp.loc[temp.storm_id.isin(storm_ids)].temp_var_accum.mean())
        spatial_atgage.append(spatial.loc[spatial.storm_id.isin(storm_ids)].spatial_var_accum.mean())

        velocity.append(vel.loc[vel.storm_id.isin(storm_ids)].velocity.mean())

        area.append(a.loc[a.storm_id.isin(storm_ids)].area.mean())
    state['temp_var_accum'] = temp_atgage
    state['spatial_var_accum'] = spatial_atgage
    state['velocity'] = velocity
    state['area'] = area

    updated_file.append(state)

# %%
########### COMBINE FILES
state_df = pd.concat(updated_file).reset_index()
state_df.to_feather('output/state')
