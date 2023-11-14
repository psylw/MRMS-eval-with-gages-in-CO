import numpy as np
from scipy.spatial import distance
from os import listdir
import math
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
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

def nearest(mrms,coord):
    xgage = [coord[i][1] for i in range(len(coord))]
    ygage =  [coord[i][0] for i in range(len(coord))]

    # find nearest coord then find surrounding 8 and label
    gage_coord = np.stack((xgage,ygage),axis=1)
    gage_coord = [tuple(gage_coord[i]) for i in range(len(gage_coord))]

    stacked = mrms.stack(coord=('longitude','latitude'))
    coord=stacked.coord.to_numpy()
    coord = [tuple(coord[i]) for i in range(len(coord))]

    min_d = []
    for i in range(len(xgage)):
        index = []
        for j in range(len(coord)):
            diff = distance.cdist([gage_coord[i]], [coord[j]],'euclidean')
            index.append(diff)
        #df = pd.DataFrame(index, columns = ['distance','MRMS','gage'])
        min_d.append(np.argmin(index))

    lat=[]
    lon=[]
    for i in min_d:
        lat.append(coord[i][1])
        lon.append(coord[i][0])

    return lat,lon

def multi_correct(mrms_rate, mrms_multi,mrms_radar):
    # divide multisesor by radar-only, fill Nan with 1
    correction = (mrms_multi/mrms_radar).fillna(1)
    # resample 1-hour to 2-min to apply to 2-min accumulation
    correction = correction.resample(time='2min').pad()
    # fill inf with 1
    correction = correction.where(correction.unknown != np.inf).fillna(1)
    # apply correction
    mrms_accum = mrms_rate*(2/60)
    mrms_2_corrected = correction*mrms_accum

    return mrms_2_corrected, correction

def gage_storm(gage):
    precip = gage['unknown']

    # create new storm id when no precip for 8hrs
    storm_id=[]
    window = timedelta(hours=6)
    storm_window = precip.rolling(window=window,min_periods=1).max()
    count = 0

    for i in range(len(storm_window)):
        if i == 0:
            storm_id.append(count) 
        elif storm_window[i]==0:
            storm_id.append(0)
        elif storm_window[i]!=0 and storm_window[i-1]==0:
            count+=1
            storm_id.append(count)
        else:
            storm_id.append(count)

    return storm_id

