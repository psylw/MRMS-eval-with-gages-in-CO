# third party imports
from os import listdir
import numpy as np
import math
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import xarray as xr

def nse(predictions, targets):
    """Takes target (gage) and prediction (radar) and calculates Nash Sutcliffe model efficiency
    coefficient.
    
    Parameters
    ----------
    predictions, targets : numpy array
        Numpy array for values of gage and radar for time period to compare.
    
    Returns
    ------
    output : float
        The NSE is computed and returned as a float.
    """
    if np.sum(targets)==0:
        nash = np.nan
    else:
        nash = (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))
    return nash


def time_change(mrms):
    """Takes mrms time (UTC) and converts to MST.
    
    Parameters
    ----------
    mrms : xarray Dataset
        Xarray Dataset with precipitation variable named "unknown" and dimensions time, latitude,
        and longitude.
    
    Returns
    ------
    output : xarray Dataset
        The time coordinate is converted to MST.
    """
    local = np.datetime_as_string(mrms.time.values, timezone=pytz.timezone('US/Mountain'))
    size = len(local[0])
    local = [local[i][:size-18] for i in range(len(local))]
    mrms = mrms.assign_coords(time=local)
    time = np.array(mrms.time.values,dtype=np.datetime64)
    mrms = mrms.assign_coords(time=time)

    return mrms

def round_coord(mrms):
    """The "round_coord" function rounds MRMS coordinates to 5th decimal place.  The raw QPE and 
    precip rate products round differely, preventing comparison of like coordinates.
    
    Parameters
    ----------
    mrms : xarray Dataset
        Xarray Dataset with precipitation variable named "unknown"and dimensions time, latitude,
        and longitude.
    
    Returns
    ------
    output : xarray DataArray
        The latitude and longitude coordinates are rounded to the 5th decimal place.
    """
    lat2 = mrms.latitude.round(5)
    lon2 = mrms.longitude.round(5)
    mrms = mrms.assign_coords({"longitude": lon2,"latitude":lat2})

    return mrms

def gage_attributes(data_folder):
    """The "gage_attributes" function finds the names and locations of gages.
    
    Parameters
    ----------
    data_folder : directory path
        Path for processed data identified in main.py.
    
    Returns
    ------
    output : Pandas DataFrame
        The locations and ID's are represented in pandas DataFrames.
    """
    # get file names from processed gage records
    filenames = listdir(data_folder +'\\gages')
    gage_id = [filenames[i][0:5] for i in range(len(filenames))]
    # read attribute table
    gage_location = pd.read_csv(data_folder +'\\Grizzly_attributes.csv')
    # drop extra data
    gage_location = gage_location.drop(columns=['Gauge_Name', 'Gauge_Name','Fire','Source',
    'Operator','Elevation','Label_Name'])
    # select gages of interest
    gage_location = gage_location[gage_location['ID'].isin(gage_id)]
    # replace longitude in datafram with global crs
    gage_location['Long'] = gage_location['Long'].to_numpy()+360
    gage_id = gage_location['ID'].values

    return gage_location, gage_id

def intensity(mrms,interval,accum):
    """The "intensity" function takes 2 minute intensity precipitation rate from MRMS and converts
     to a 15, 30, and 60 minute intensity.
    
    Parameters
    ----------
    mrms : xarray Dataset
        Xarray Dataset with precipitation rate variable named "unknown"

    interval : int or float
        Interval for intensity in minutes

    accum : xarray Dataset
        2-min accumulation calulated from either radar-only 2-min precipitaion rate or multisensor
        2-min precipitaion
    
    Returns
    ------
    output : xarray DataArray
        The precipitation variable "unknown" is converted to desired intensity interval.
    """
    # resample to match gage frequency, value is filled every 2 min
    accum = accum.unknown.resample(time='1min').asfreq()
    accum = accum.fillna(0)
    # calculate desired intensity by summing 1min accum with rolling desired intensity window, 
    # then divide by desired intensity (min) and convert to mm/hr
    mrms = (accum.rolling(time=interval,min_periods=1).sum())*(60/interval)

    return mrms

def open_gage(intensity, gage_id, data_folder):
    """The "open_gage" function opens the desired precipitation variables from the processed gage
    data.
    
    Parameters
    ----------
    intensity : float int or float
        Interval for intensity in minutes or string specifying desired variable is accumulation
    
    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    data_folder : directory path
        Path for processed data identified in main.py

    Returns
    ------
    output : pandas DataFrame
        Precipitation rate for specified intensity for all gages or precipitation accumulation.
    """
    gage = {}
    j=0
    # if else statement to get either total accumulation or intensity
    if type(intensity) == str:
        for i in gage_id:
            gage[i] = pd.read_csv(data_folder +'\\gages'+'\\'+i+'_20210727_20210930_checked_processed.csv')
            gage[i]['TimeStamp (Local)'] = pd.to_datetime(gage[i]['TimeStamp (Local)'])
            gage[i]= gage[i].set_index("TimeStamp (Local)")
            gage[i] = gage[i]['Bin Accum (mm)']
            gage[i] = gage[i].fillna(0)
            j+=1
    else:
        for i in gage_id:
            gage[i] = pd.read_csv(data_folder +'\\gages'+'\\'+i+'_20210727_20210930_checked_processed.csv')
            gage[i]['TimeStamp (Local)'] = pd.to_datetime(gage[i]['TimeStamp (Local)'])
            gage[i]= gage[i].set_index("TimeStamp (Local)")
            gage[i] = gage[i][str(intensity)+'-minute Intensity (mm/h)']
            gage[i] = gage[i].fillna(0)
            j+=1

    gage = pd.concat(gage, axis = 1)
    gage = gage.fillna(0)
    return gage

def storm(gage_accum):
    """The "storm" function opens the precipitation accumulation variable from the processed gage
    data and identifies individual storms.  Storms are identified based on recordings from all
    gages.  A new event begins following a period of at least 8 h with no precipitation among the 
    gages. Rain-free periods of less than 8 h are included within an encompassing event.
    
    Parameters
    ----------
    gage_accum : Pandas DataFrame
        Precipitation accumulation for all gages

    Returns
    ------
    output : pandas DataFrame
        DataFrame with storm_id, start time, end time, and duration
    """    

    gage = gage_accum
    # create boolean value for rain, true if any gage sees rain
    rain = []
    for i in range(len(gage)):
        rain.append((gage.values[i]>0).any())
    gage['rain']=rain
    # create storm id consistent among all gages
    count=0
    storm_id=[]
    first = gage.index[0]+timedelta(hours=8)
    new_storm = []

    for i in range(len(gage)):
        #make window of 8 hours
        if gage.index[i]<=first:
            start = gage.index.get_loc(gage.index[0])
            stop = gage.index.get_loc(gage.index[i])
        else:
            start = gage.index.get_loc(gage.index[i]-timedelta(hours=8))
            stop = gage.index.get_loc(gage.index[i])
        window = gage.iloc[start:(stop+1)]['rain']

        test = any(window)

        if test == True:
            storm_id.append(count)
            new_storm.append(test)
        else:
            new_storm.append(test)
            storm_id.append(math.nan)
            if new_storm[i-1]==True:
                count+=1
    gage['storm_id']=storm_id
    # get start and end of storms
    storm_time = []
    for i in gage.dropna()['storm_id'].unique():
        start= (gage.loc[gage['storm_id']==i].index[0].round('60min'))
        end = (gage.loc[gage['storm_id']==i].index[-1].round('60min'))
        storm_time.append([i,start,end])
    storm_time = pd.DataFrame(storm_time, columns = ['storm_id','start','end'])
    storm_time['length'] = storm_time['end']-storm_time['start']

    return storm_time

def fill(gage,storm_time):  
    """The "fill" function fills the gage precipitation variables with 0 where there is no data or
    the value is nan, making all variables the same length.
    
    Parameters
    ----------

    gage : pandas DataFrame
        Precipitation rate for specified intensity for all gages or precipitation accumulation
    
    Returns
    ------
    output : pandas DataFrame
        Precipitation variables with equal length
    """

    start,end = storm_time['start'][0],storm_time['end'].iloc[-1] 
    rng = pd.date_range(start, end, freq='1min') 
    gage = gage.reindex(rng,fill_value=0)

    return gage

def nearest(mrms, gage_location, gage_id):
    """The "nearest" function finds the closest MRMS coordinate to the gage, using euclidean
    distance, and the neighboring 8 coordinates.
    
    Parameters
    ----------

    mrms : xarray Dataset
        Xarray Dataset with precipitation rate variable named "unknown"

    gage_location : Pandas DataFrame
        A pandas DataFrames with the gage locations.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    Returns
    ------
    output : pandas DataFrame
        Outputs a dataframe with the closest coordinate and nearest 8 coordinates for each gage.
    """
    xgage = gage_location['Long'].values
    ygage = gage_location['Lat'].values

    # find nearest coord then find surrounding 8 and label
    gage_coord = np.stack((xgage,ygage),axis=1)
    gage_coord = [tuple(gage_coord[i]) for i in range(len(gage_coord))]

    stacked = mrms.stack(coord=('longitude','latitude'))
    coord=stacked.coord.to_numpy()
    coord = [tuple(coord[i]) for i in range(len(coord))]

    index = []

    for i in range(len(xgage)):
        for j in range(len(coord)):
            diff = distance.cdist([gage_coord[i]], [coord[j]],'euclidean')
            index.append([diff,coord[j],gage_coord[i]])

    df = pd.DataFrame(index, columns = ['distance','MRMS','gage'])

    center=[]
    N=[]
    S=[]
    E=[]
    W=[]
    NE=[]
    NW=[]
    SE=[]
    SW=[]

    for i in range(6):
        gage_loc = df.loc[df['gage']==gage_coord[i]]
        sort = gage_loc['distance'].sort_values(ascending=True)
        index_9 = sort[0:1]
        C = df['MRMS'].iloc[index_9.index].values
        center.append(np.round(C[0],5))
        # now find surrounding 8
        x=C[0][0]
        y=C[0][1]
        diffx = mrms.longitude[1].values-mrms.longitude[0].values
        diffy = mrms.latitude[1].values-mrms.latitude[0].values
        N.append(np.round(tuple([x,(y+diffy)]),5))
        S.append(np.round(tuple([x,(y-diffy)]),5))
        E.append(np.round(tuple([(x+diffx),y]),5))
        W.append(np.round(tuple([(x-diffx),y]),5))
        NE.append(np.round(tuple([(x+diffx),(y+diffy)]),5))
        NW.append(np.round(tuple([(x-diffx),(y+diffy)]),5))
        SE.append(np.round(tuple([(x+diffx),(y-diffy)]),5))
        SW.append(np.round(tuple([(x-diffx),(y-diffy)]),5))

    closest = pd.DataFrame(gage_id,columns=['gage_id'])
    closest['Center'] = center
    closest['North']=N
    closest['South']=S
    closest['East']=E
    closest['West']=W
    closest['NorthEast']=NE
    closest['NorthWest']=NW
    closest['SouthEast']=SE
    closest['SouthWest']=SW
    
    return closest

def mrms_nearest(mrms,closest):
    """The "mrms_nearest" function takes the closest coordinates and nearest 8 coordinates for 
    each gage and selects the data from the MRMS dataset at those locations.
    
    Parameters
    ----------
    mrms : xarray Dataset
        Xarray Dataset with precipitation rate variable named "unknown"
    
    closest : pandas Dataframe
        Outputs a dataframe with the closest coordinate and nearest 8 coordinates for each gage
    
    Returns
    ------
    output : list of xarray DataArray
        The coordinates for each gage, and associated precipitation variables, are selected from the
    MRMS dataset.  Each gage is a separate DataArray, and the output is a list of DataArrays.
        
    """
    
    mrms_ATgage=[]

    for i in range(6):
        x=[]
        y=[]
        for j in np.arange(1,10,1):
            x.append(closest.iloc[i][j][0])
            y.append(closest.iloc[i][j][1])
            
        mrms_ATgage.append(mrms.sel(longitude=x,latitude=y))
    
    return mrms_ATgage

def multi_correct(mrms_radar, mrms_multi, mrms_2):
    """The "multi_correct" function calculates a spatially and temporally varying corrected factor
    by comparing the radar only QPE to the multisensor Pass 2 QPE.  The correction is then applied 
    to the 2-min accumulation, calculated frm the 2-min precipitation rate product.
    
    Parameters
    ----------

    mrms_radar : xarray Dataset
        Radar only 1-hour QPE.

    mrms_multi : xarray Dataset
        Pass 2 multisensor 1-hour QPE

    mrms_2 : xarray Dataset
        Precipitation accumulation in 2-min bins, calculated from the 2-min precipitation rate.
    
    Returns
    ------
    output : xarray Dataset
        The corrected 2-min accumulation and corrected are outputed as xarray Datasets.
        
    """
    # divide multisesor by radar-only, fill Nan with 1
    correction = (mrms_multi/mrms_radar).fillna(1)
    # resample 1-hour to 2-min to apply to 2-min accumulation
    correction = correction.resample(time='2min').pad()
    # fill inf with 1
    correction = correction.where(correction.unknown != np.inf).fillna(1)
    # apply correction
    mrms_2_corrected = correction*mrms_2

    return mrms_2_corrected, correction

def bias(mrms_int, gage_int, gage_id,storm_time):
    """The "bias" function calculates the bias as a function of maximum MRMS intensity vs maximum
    gage intensity.
    
    Parameters
    ----------

    mrms_int : xarray Dataset
        Xarray Dataset with precipitation rate variable named "unknown"

    gage_int : Pandas DataFrame
        Pandas dataframe with with the precipitation rate from the gage.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : pandas DataFrame
        Pandas dataframe with bias and max intensities for each gages and surrounding coordinates
        per storm.
        
    """
    count =0
    max_perstorm = []
    for j in gage_id:   
        for i in range(9): 
            for k in range(len(storm_time)): 
                start = storm_time['start'][k]
                end = storm_time['end'][k]

                # select times in storm and all 9 coordinates, find max values
                m_max = mrms_int[count][:,i,i].sel(time=slice(start,end)).values.max()
                g_max = gage_int[j][start:end].max()

                # find time of max
                g_max_time = gage_int[j][start:end].idxmax()
                # find time of max
                mint = mrms_int[count][:,i,i].sel(time=slice(start,end))
                m_max_time = mint.loc[mint==m_max].time.values

                if g_max!=0:
                    bias = m_max/g_max
                else:
                    bias = np.nan
                max_perstorm.append([j,i,k,bias,m_max,g_max,g_max_time,m_max_time])
        count+=1
        
    max_perstorm = pd.DataFrame (max_perstorm, columns = ['gage_id','cell_id','storm_id','bias','max_mrms',
                                                          'max_gage','gage_time','mrms_time'])   
    max_perstorm['mrms_time'] = [max_perstorm['mrms_time'][i][0] for i in range(len(max_perstorm))]
        
    return max_perstorm

def mean(mrms, gage_id):
    """The "mean" function calculates the mean of the closest and surrounding coordinates from the
    MRMS dataset.
    
    Parameters
    ----------

    mrms : xarray Datarray
        Xarray Dataset with precipitation intensity or accumulation at the gages.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    Returns
    ------
    output : xarray Datarray
        Mean value of closest and surrounding coordinates at each gage.
        
    """
    mrms_mean = []
    for i in range(len(gage_id)):
        mrms_mean.append(mrms[i].mean(dim=["latitude","longitude"]))
    
    return mrms_mean

def max_cells(mrms, gage_id):
    """The "max_cells" function calculates the max of the closest and surrounding coordinates from the
    MRMS dataset.
    
    Parameters
    ----------

    mrms : xarray Datarray
        Xarray Dataset with precipitation intensity or accumulation at the gages.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    Returns
    ------
    output : xarray Datarray
        Max value of closest and surrounding coordinates at each gage.
        
    """
    mrms_max = []
    for i in range(len(gage_id)):
        mrms_max.append(mrms[i].max(dim=["latitude","longitude"]))
    
    return mrms_max

def median(mrms, gage_id):
    """The "median_cells" function calculates the median of the closest and surrounding coordinates from the
    MRMS dataset.
    
    Parameters
    ----------

    mrms : xarray Datarray
        Xarray Dataset with precipitation intensity or accumulation at the gages.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    Returns
    ------
    output : xarray Datarray
        Median value of closest and surrounding coordinates at each gage.
        
    """
    mrms_median = []
    for i in range(len(gage_id)):
        mrms_median.append(mrms[i].median(dim=["latitude","longitude"]))
    
    return mrms_median

def best(mrms, gage, gage_id):
    """The "best" function calculates the absolute error at each timestep between the gage and MRMS
    precipitation variables.  For each timestep, the best coordinate, with the lowest error, is 
    identified.
    
    Parameters
    ----------

    mrms : xarray Dataset
        Xarray Dataset with precipitation rate variable named "unknown"

    gage : Pandas DataFrame
        A pandas DataFrame with the precipitation rate for each gage      

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages
    
    Returns
    ------
    output : pandas DataFrame
        Returns a pandas dataframe with the coordate ID with the lowest error at each timestep, for 
    each gage.
        
    """
    # find error, for each gage, at every timestep
    error = []
    for i in range(len(gage_id)):
        for j in range(9):
            g = gage[gage_id[i]]
            m = mrms[i][:,j,j]   
            error.append(np.abs(g-m))

    error = pd.concat(error,axis=1)

    # find cell with lowest error for each timestep at each gage
    low = []
    for j in gage_id:
        for i in range(len(error)):
            temp = pd.DataFrame(error.iloc[i][j])
            if temp.duplicated().any()==True:
                duplicated=True
            elif temp.duplicated().any()!=True:
                duplicated=False
            min_index = temp.set_index(np.arange(0,9,1)).idxmin().values
            min_error = temp.min().values
            low.append([error.index[i],j,min_index,min_error,duplicated])
            
    low = [[low[i][0],low[i][1],low[i][2][0],low[i][3][0],low[i][4]] for i in range(len(low))]
    
    # append values to new dataframe
    low = pd.DataFrame(low,columns = ['time','gage','cell','error','duplicated'])
    best_cell = pd.DataFrame(index=gage.index)
    
    for i in gage_id:
        best_cell[i+'error']=low.loc[low['gage']==i]['error'].values
        best_cell[i+'cell']=low.loc[low['gage']==i]['cell'].values
        best_cell[i+'dupicated']=low.loc[low['gage']==i]['duplicated'].values
    return best_cell

def total_accum(gage,mrms,storm_time,gage_id):
    """The "total_accum" function calculates the total accumulation for each storm for gages and 
    MRMS.
    
    Parameters
    ----------
    gage : Pandas DataFrame
        Pandas dataframe with with the precipitation accumulation from the gage.

    mrms : xarray Dataset
        Xarray Dataset with 1-hour multisensor QPE

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.    

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    Returns
    ------
    output : pandas DataFrame and xarray DataArray
        Storm total accumulation for gages and MRMS.
        
    """
    gage_accum = []
    mrms_accum = []
    mrms_time = []
    for i in range(len(storm_time)):
        start = storm_time['start'][i]
        end = storm_time['end'][i]
        m = mrms.sel(time=slice(start,end)).cumsum(dim='time',keep_attrs=True)
        mrms_time.append(mrms.sel(time=slice(start,end)).time)
        g = gage[gage_id][start:end].cumsum()
        mrms_accum.append(m)
        gage_accum.append(g)

    mrms_accum = xr.concat(mrms_accum,dim='time')
    gage_accum = pd.concat(gage_accum,axis=0)
    mrms_time = xr.concat(mrms_time,dim='time').values
    mrms_accum = mrms_accum.assign_coords(time=mrms_time)
    return mrms_accum,gage_accum

def best_cell_accum(bias_accum,gage_id,storm_time):
    """The "best_cell_accum" function uses the absolute error between the gage and MRMS storm total
    accumulation to identify the coordinate that performs the best for each storm.
    
    Parameters
    ----------

    bias_accum : pandas dataframe
        Pandas dataframe with storm total accumulation bias

    gage : Pandas DataFrame
        A pandas DataFrame with the precipitation rate for each gage      

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages
    
    Returns
    ------
    output : pandas DataFrame
        Returns a pandas dataframe with the coordate ID with the lowest error for each storm, for 
    each gage.  When there are duplicate error values, the function defaults to the lowest index.
    Need a second dataframe that replaces duplicates with Nan's to see unbiased performance of each
    coordinate.
        
    """
    best_cell_accum = []
    bias_accum['error'] = (bias_accum['max_gage']-bias_accum['max_mrms']).abs()
    for i in range(len(gage_id)):
        for k in storm_time['storm_id']:
            index = bias_accum.loc[(bias_accum['gage_id']==gage_id[i])&(bias_accum['storm_id']==k)]['error'].idxmin()
            cell = bias_accum.iloc[index]['cell_id']
            best_cell_accum.append(bias_accum.loc[(bias_accum['gage_id']==gage_id[i])&(bias_accum['storm_id']==k)&(bias_accum['cell_id']==cell)])
    best_cell_accum = pd.concat(best_cell_accum,axis=0)
    
    return best_cell_accum