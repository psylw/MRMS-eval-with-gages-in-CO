# third party imports
import os
from os import listdir
import numpy as np
import math
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import mstats

import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
from matplotlib import cm

from scipy.spatial.distance import cdist
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from functions import nse, mean

import geopandas as gpd
import rioxarray as rxr

def allcells_gagevmrms(mrms, gage, title, gage_id, storm_time,closest):
    """The "allcells_gagevmrms" function plots gage vs mrms for each coordinate (9 plots total).
    
    Parameters
    ----------
    mrms : xarray Dataset
        Dataset with MRMS precip rate at gage.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.

    title : string
        Title for plot

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.

    closest : pandas Dataframe
        Outputs a dataframe with the closest coordinate and nearest 8 coordinates for each gage
        
    Returns
    ------
    output : subplot
        Subplot with 9 frames.
    """
    fig, axs = plt.subplots(9,1, figsize=(20, 40), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .4, wspace=0)
    fig.suptitle(title,y=0.9,size=20)
    axs = axs.ravel()
    count = 0

    for k in range(9):
        g = []
        #u = []
        c = []
        for i in range(len(gage_id)):

            for j in storm_time.index:
                start = storm_time['start'][j]
                end = storm_time['end'][j]

                gage_storm = gage[start:end][gage_id[i]]
                mrms_storm = mrms[i].sel(time=slice(start,end))[:,k,k]
                
                axs[count].scatter(gage_storm,mrms_storm,c='r',marker='^')             
                
                x1 = (0,np.max(gage_storm))
                y1 = (0,np.max(gage_storm))
                axs[count].plot(x1,y1,'g')
                axs[count].set_title(closest.columns[k+1])
                axs[count].set_ylabel('MRMS (mm/hr)',size=14)
                axs[count].set_xlabel('gage (mm/hr)',size=14)
                g.append(gage_storm.values)
                c.append(mrms_storm.values)
        g = np.concatenate(g)
        c = np.concatenate(c)
        rmse_c = mean_squared_error(g, c, squared=False)
        nse_c = nse(c, g)
        textstr = '\n'.join((
        r'$rmse=%.2f$' % (rmse_c, ),
        r'$nse=%.2f$' % (nse_c, )))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.9, 0.8, textstr, transform=axs[count].transAxes,fontsize=14,verticalalignment='top',bbox=props)
        count+=1

    return 

def best_gagevmrms(mrms,gage,title,best_cell, gage_id, storm_time):
    """The "best_gagevmrms" function plots gage vs "best" multisensor corrected MRMS for all values.
    
    Parameters
    ----------
    mrms : xarray Dataset
        Dataset with MRMS precip rate at gage.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.

    title : string
        Title for plot

    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : plot
        Plot of all gage and MRMS values.
    """
    fig,ax = plt.subplots(figsize=(20, 10), facecolor='w', edgecolor='k')
    g = []
    c = []
    for i in range(len(gage_id)):
        for j in storm_time.index:
            start = storm_time['start'][j]
            end = storm_time['end'][j]

            gage_storm = gage[start:end][gage_id[i]]
            cell = best_cell[start:end][gage_id[i]+'cell']
            mrms_C_storm = mrms[i].sel(time=slice(start,end))
            
            C = []
            n=0
            for k in cell:
                C.append(mrms_C_storm[n,k,k].values)
                n+=1
            g.append(gage_storm.values)
            c.append(C)
            plt.scatter(gage_storm,C,c='r',marker='^')

            x1 = (0,np.max(gage_storm))
            y1 = (0,np.max(gage_storm))
            plt.plot(x1,y1,'g')
            
            plt.ylabel('MRMS (mm/hr)',size=14)
            plt.xlabel('gage (mm/hr)',size=14)
            
    g = np.concatenate(g)
    c = np.concatenate(c)

    nash = nse(c,g)
    textstr = r'$nse=%.2f$' % (nash, )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.9, 0.8, textstr, transform=ax.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    plt.title(title,y=.95,size=20)

    return 

def best_errorvcorrection(gage, mrms, title, best_cell,correction,gage_id,storm_time):
    """The "best_errorvcorrection" function plots the multisensor correction vs the error between 
    gage and MRMS values.
    
    Parameters
    ----------
    mrms : xarray Dataset
        Dataset with MRMS precip rate at gage.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.

    title : string
        Title for plot

    correction : xarray Dataset
        Multisensor correction

    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : plot
        Plot of the error vs multisensor correction.
    """
    fig = plt.subplots(figsize=(20, 10), facecolor='w', edgecolor='k')
    plt.title(title,y=0.95,size=20)
    for i in range(len(gage_id)):
        for j in storm_time.index:
            start = storm_time['start'][j]
            end = storm_time['end'][j]
            gage_storm = gage[start:end][gage_id[i]]
            cell = best_cell[start:end][gage_id[i]+'cell']
            mrms_C_storm = mrms[i].sel(time=slice(start,end))
            
            C = []
            n=0
            for k in cell:
                C.append(mrms_C_storm[n,k,k].values)
                n+=1
                
            error = gage_storm - C
            correction_storm = correction[i].sel(time=slice(start,end))
            
            cor = []
            n=0
            for k in cell:
                cor.append(correction_storm.unknown[n,k,k].values)
                n+=1
            plt.scatter(cor,error,c='r',marker='^')
            plt.ylabel('Gage - MRMS(mm/hr)',size=14)
            plt.xlabel('multisensor correction',size=14)
            plt.xscale('log')

    return 

def gagevmrms_substorm(mrms,gage,best_cell,gage_id,storm_time):
    """The "gagevmrms_substorm" function plots the gage vs MRMS values for each storm.  Gages have 
    different colors.
    
    Parameters
    ----------

    mrms : xarray Dataset
        Dataset with MRMS precip rate at gage.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.

    title : string
        Title for plot

    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : subplot
        Subplot of gage vs mrms for each storm.
    """
    fig, axs = plt.subplots(29,1, figsize=(20, 290), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .3, wspace=0.1)
    #fig.suptitle(title,y=0.5,size=20)
    axs = axs.ravel()
    count = 0

    for j in storm_time.index:
        x = []
        y = []

        for i in range(len(gage_id)):    
            start = storm_time['start'][j]
            end = storm_time['end'][j]
            gage_storm = gage[start:end][gage_id[i]]
            cell = best_cell[start:end][gage_id[i]+'cell']
            mrms_C_storm = mrms[i].sel(time=slice(start,end))
            
            C = []
            n=0
            for k in cell:
                C.append(mrms_C_storm[n,k,k].values)
                n+=1
                
            axs[count].scatter(gage_storm,C)
            x.append(gage_storm)
            y.append(C)
            axs[count].set_title('Storm ID: '+str(j)+', Storm Start: '+str(storm_time.iloc[j]['start'])+', Storm End: '+str(storm_time.iloc[j]['end']))
            axs[count].legend(['GCFC2', 'GCEC2', 'GCNC2', 'GCTC2', 'GCCC2', 'GCDC2'],loc='upper right',fontsize = 14)
        x1 = (0,np.max(x))
        y1 = (0,np.max(x))
        axs[count].plot(x1,y1,'g')
        count+=1

    return

def gagevmrms_substorm_subgage(mrms,gage,title,best_cell,gage_id,storm_time):
    """The "gagevmrms_substorm_subgage" function plots the gage vs MRMS values for each storm and
    gage.
    
    Parameters
    ----------
    mrms : xarray Dataset
        Dataset with MRMS precip rate at gage.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.

    title : string
        Title for plot

    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : subplot
        Subplot of gage vs mrms for each storm and gage.
    """
    fig, axs = plt.subplots(29,6, figsize=(60, 200), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .3, wspace=0.1)
    fig.suptitle(title,y=0.89,size=20)
    axs = axs.ravel()
    count = 0

    for j in storm_time.index:
        for i in range(len(gage_id)):    
            start = storm_time['start'][j]
            end = storm_time['end'][j]

            gage_storm = gage[start:end][gage_id[i]]
            
            cell = best_cell[start:end][gage_id[i]+'cell']
            mrms_C_storm = mrms[i].sel(time=slice(start,end))
            
            C = []
            n=0
            for k in cell:
                C.append(mrms_C_storm[n,k,k].values)
                n+=1

            axs[count].scatter(gage_storm,C)

            axs[count].set_title('Gage ID:'+gage_id[i]+', Storm ID: '+str(j))
            x1 = (0,np.max(gage_storm))
            y1 = (0,np.max(gage_storm))
            axs[count].plot(x1,y1,'g')
            axs[count].legend(['rmse = '+str(mean_squared_error(gage_storm, C, squared=False))],loc='upper right',fontsize = 14)
            count+=1
    return

def timeseries_int(mrms,gage,title,best_cell,gage_id,storm_time):
    """The "timeseries" function plots the gage and best MRMS values as a function of time for each
    storm.  The best coordinate is indexed differently for intensity and accumulation, so there's a
    different function for each.
    
    Parameters
    ----------
    mrms : xarray Dataset
        Dataset with MRMS precip rate at gage.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.

    title : string
        Title for plot

    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : subplot
        Timeseries subplot for each storm and gage.
    """
    fig, axs = plt.subplots(29,6, figsize=(60, 200), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .3, wspace=0.1)
    fig.suptitle(title,y=0.89,size=20)
    axs = axs.ravel()
    count = 0

    for j in storm_time.index:
        for i in range(len(gage_id)):    
            start = storm_time['start'][j]
            end = storm_time['end'][j]

            gage_storm = gage[start:end][gage_id[i]]
            axs[count].plot(gage_storm.index,gage_storm)
            
            cell = best_cell[start:end][gage_id[i]+'cell']
            mrms_C_storm = mrms[i].sel(time=slice(start,end))
            
            C = []
            n=0
            for k in cell:
                C.append(mrms_C_storm[n,k,k].values)
                n+=1
            
            rmse = mean_squared_error(gage_storm.values,C,squared=False)
            nash = nse(C,gage_storm.values)
            textstr = '\n'.join((
                r'$rmse=%.2f$' % (rmse, ),
                r'$nse=%.2f$' % (nash, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[count].text(0.9, 0.8, textstr,transform=axs[count].transAxes,fontsize=14,verticalalignment='top',bbox=props)
            axs[count].plot(mrms_C_storm.time,C)
            axs[count].set_title('Gage ID:'+gage_id[i]+', Storm ID: '+str(j))
            axs[count].legend(['Gage','MRMS_C','MRMS_U'],loc='upper right',fontsize = 14)

            count+=1
    return

def bestcell_bystorm(best_cell,gage_id,storm_time,closest):
    """The "bestcell_bystorm" function plots a heatmap of the probability of each coordinate for 
    each gage and storm.
    
    Parameters
    ----------
    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.

    closest : pandas Dataframe
        Outputs a dataframe with the closest coordinate and nearest 8 coordinates for each gage
    
    Returns
    ------
    output : subplot
        Subplot with a heatmap for each gage and storm.
    """
    # probability of cell being best per storm
    prob_storm = []
    index = []
    cells = closest.columns[1:10].values
    for i in range(len(gage_id)):
        for j in storm_time.index:
            start = storm_time['start'][j]
            end = storm_time['end'][j]
            best_storm = best_cell[start:end]
            best = best_storm.loc[best_storm[gage_id[i]+'dupicated']==False]
            best_c = best[gage_id[i]+'cell'].values

            n,bins = np.histogram(best_c,bins=np.arange(0,10,1))
            probability = n/len(best_c)

            prob_storm.append([[j]*9,[gage_id[i]]*9,cells,probability,n])

    prob_storm = [pd.DataFrame(prob_storm[i]).T for i in range(len(prob_storm))]
    prob_storm = pd.concat(prob_storm)
    prob_storm = prob_storm.rename(columns={0:'storm_id',1:'gage_id',2:'cell_id',3:'probability',4:'count'})
    
    for i in storm_time.index:
        for j in gage_id:
            
            df = prob_storm.loc[(prob_storm['storm_id']==i)&(prob_storm['gage_id']==j)]['probability']
            probability = [df.iloc[[6,1,5]].values.tolist(),df.iloc[[4,0,3]].values.tolist(),df.iloc[[8,2,7]].values.tolist()]

            im = plt.imshow(probability, cmap = 'Greens')
            #plt.colorbar(im,ax=axs[k])
            plt.xticks([])
            plt.yticks([])
            plt.title ('Gage id: '+j+',Storm id: '+str(i))

            plt.show()
    return

def timeseries_accum(mrms,gage,title,best_cell,gage_id,storm_time):
    """The "timeseries_accum" function plots the gage and best MRMS values as a function of time for
    each storm.
    
    Parameters
    ----------
    mrms : xarray Dataset
        Dataset with MRMS accumulation at gage.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.

    title : string
        Title for plot

    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.

    Returns
    ------
    output : subplot
        Timeseries subplot for each storm and gage.
    """
    fig, axs = plt.subplots(29,6, figsize=(60, 200), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .3, wspace=0.1)
    fig.suptitle(title,y=0.89,size=20)
    axs = axs.ravel()
    count = 0

       # for each gage
    for j in storm_time.index:
        for i in range(len(gage_id)):    
            start = storm_time['start'][j]
            end = storm_time['end'][j]

            gage_storm = gage[start:end][gage_id[i]]
            axs[count].plot(gage_storm.index,gage_storm)
            
            cell = best_cell.loc[(best_cell['storm_id']==j)&(best_cell['gage_id']==gage_id[i])]['cell_id'].values[0]
            mrms_storm = mrms[i].sel(time=slice(start,end))
            
            axs[count].plot(mrms_storm.time,mrms_storm[:,cell,cell])
            axs[count].set_title('Gage ID:'+gage_id[i]+', Storm ID: '+str(j))

            count+=1
    return

def bias_storm(bias, title, gage_id, storm_time):
    """The "bias_storm" function plots the total accumulation bias or the maximum intensity bias for
    each storm.
    
    Parameters
    ----------
    bias : pandas DataFrame
        Either total accumulation bias or max intensity bias

    title : string
        Title for plot

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.

    Returns
    ------
    output : plot
        Plot of storm vs bias at all gages.
    """
    fig = plt.subplots(figsize=(20, 10), facecolor='w', edgecolor='k')
    bias['bias_opt'] = np.abs(1-bias['bias'])
    plt.title(title,y=0.89,size=20)
    plt.xlabel('Storm_id',size=14)
    plt.ylabel('max MRMS/max gage',size=14)
    n = 6
    color = cm.rainbow(np.linspace(0, 1, n))
    plt.xticks(np.arange(storm_time['storm_id'].min(), storm_time['storm_id'].max()+1, 1.0))

    for i in storm_time['storm_id']:
        for j in range(len(gage_id)):
            c = color[j]
            best_bias = bias.loc[(bias['storm_id']==i)&(bias['gage_id']==gage_id[j])]['bias_opt'].min()
            plot_bias = bias.loc[(bias['storm_id']==i)&(bias['gage_id']==gage_id[j])&(bias['bias_opt']==best_bias)].min()['bias']
            plt.scatter(i,plot_bias,color=c)
    plt.legend(['GCFC2', 'GCEC2', 'GCNC2', 'GCTC2', 'GCCC2', 'GCDC2'])
    return

def group_best(mrms,gage,mean,median,max_cells, best_cell,title, gage_id, storm_time):
    """The "group_best" function plots gage vs "best", mean, median, and max of multisensor corrected MRMS for all values.
    
    Parameters
    ----------
    mrms : xarray Dataset
        DataArray for intensity or accumulation for the cell with the lowest error.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.
        
    mean : xarray Dataset
        DataArray for mean of cells

    median : xarray Dataset
        DataArray for median of cells
        
    max_cells : xarray Dataset
        DataArray for max of cells
        
    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    title : string
        Title for plot

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : plot
        Plot of all gage and MRMS values for best, mean, median, and max of cells.
    """
    
    # gage vs mean,median, max of coords
    fig,ax = plt.subplots(figsize=(20, 10), facecolor='w', edgecolor='k')
    g = []
    c = []
    mean_s = []
    median_s = []
    max_s = []
    n = 4
    color = cm.rainbow(np.linspace(0, 1, n))

    for i in range(len(gage_id)):
        for j in storm_time.index:
            start = storm_time['start'][j]
            end = storm_time['end'][j]

            gage_storm = gage[start:end][gage_id[i]]
            mean_storm = mean[i].sel(time=slice(start,end))
            median_storm = median[i].sel(time=slice(start,end))
            max_storm = max_cells[i].sel(time=slice(start,end))
            cell = best_cell[start:end][gage_id[i]+'cell']

            mrms_C_storm = mrms[i].sel(time=slice(start,end))
            C = []
            n=0
            for k in cell:
                C.append(mrms_C_storm[n,k,k].values)
                n+=1
            g.append(gage_storm.values)
            c.append(C)
            mean_s.append(mean_storm.values)
            median_s.append(median_storm.values)
            max_s.append(max_storm.values)

            plt.scatter(gage_storm,C,color=color[0],marker='^')
            plt.scatter(gage_storm,mean_storm,color=color[1],marker='x')
            plt.scatter(gage_storm,median_storm,color=color[2],marker='o')
            plt.scatter(gage_storm,max_storm,color=color[3],marker='*')

            x1 = (0,np.max(gage_storm))
            y1 = (0,np.max(gage_storm))
            plt.plot(x1,y1,'g')

            plt.ylabel('MRMS (mm/hr)',size=14)
            plt.xlabel('gage (mm/hr)',size=14)
            plt.legend(['best','mean','median','max'])

    g = np.concatenate(g)
    c = np.concatenate(c)
    mean_s = np.concatenate(mean_s)
    median_s = np.concatenate(median_s)
    max_s = np.concatenate(max_s)

    rmse_best = mean_squared_error(g, c, squared=False)
    rmse_mean = mean_squared_error(g, mean_s, squared=False)
    rmse_median = mean_squared_error(g, median_s, squared=False)
    rmse_max = mean_squared_error(g, max_s, squared=False)

    textstr = '\n'.join((
        r'$rmse_best=%.2f$' % (rmse_best, ),
        r'$rmse_mean=%.2f$' % (rmse_mean, ),
        r'$rmse_median=%.2f$' % (rmse_median, ),
        r'$rmse_max=%.2f$' % (rmse_max, )))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.9, 0.8, textstr, transform=ax.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    plt.title(title,y=.95,size=20)

    return

def animation_gages(mrms,gage_location,closest,storm_time):
    """The "animation_gages" function creates an animation of the 1hr QPE that is focused on the gages.
    
    Parameters
    ----------
    mrms : xarray Dataset
        DataArray for multisensor QPE for all of Colorado.

    gage_location : pandas Dataframe
        Dataframe with gage coordinates

    closest : pandas Dataframe
        Dataframe with closest 9 coordinates of MRMS 

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : animation
        Animation of 1hr QPE for each storm with gage locations.
    """
    # animation
    # things to add: topography, wind, pizza slices

    mrms = mrms.unknown.where(mrms.unknown>0)

    lon, lat = np.meshgrid(mrms.longitude,mrms.latitude)

    fig = plt.figure(1, figsize=(20,16))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                           hspace=0.01, wspace=0.01)

    plotcrs = ccrs.LambertConformal(central_latitude=38.5, central_longitude=-105)
    ax = plt.subplot(1,1,1, projection=plotcrs)
    #ax.set_extent((252.3, 253.5, 39.2, 40)) # slightly larger area
    ax.set_extent((252.685, 252.885, 39.535, 39.665)) # focus on gages
    x = gage_location['Long']
    y = gage_location['Lat']
    ax.scatter(x,y,s=60,c='r',transform=ccrs.PlateCarree())
    closest_x=[]
    closest_y=[]
    for i in range(6):
        for j in np.arange(1,10,1):
            closest_x.append(closest.iloc[i][j][0])
            closest_y.append(closest.iloc[i][j][1])
    ax.scatter(closest_x,closest_y,s=60,c='b',transform=ccrs.PlateCarree())

    for i in gage_location.index:
        ax.text(x[i],y[i],gage_location['ID'][i],transform=ccrs.PlateCarree(), fontsize=20,color='r')

    plt.rcParams['animation.html']='jshtml'
    artists = []
    ds = []
    for j in storm_time.index:
        start = storm_time['start'][j]
        end = storm_time['end'][j]
        mrms_storm = mrms.sel(time=slice(start,end))    

        for i in range(len(mrms_storm)):
            ds = mrms_storm[i]
            this_time = 'Storm_ID: '+str(storm_time.iloc[j]['storm_id'])+', Storm Date: '+str(mrms_storm[i].time.values)[0:10]+', Hour: '+str(mrms_storm[i].time.values)[11:16]
            text = ax.text(0.5,1,this_time,ha='center',verticalalignment='bottom',transform=ax.transAxes, fontsize=20)
            mesh = ax.pcolormesh(lon,lat,ds,transform=ccrs.PlateCarree(),cmap='viridis', alpha=0.6,vmin=0,vmax=24)    
            artists.append([mesh,text])

    anim = ArtistAnimation(fig, artists, interval = 200)
    return anim

def animation_larger(mrms,gage_location,closest,storm_time):
    """The "animation_larger" function creates an animation of the 1hr QPE for the area
    around the Grizzly creek burn area.
    
    Parameters
    ----------
    mrms : xarray Dataset
        DataArray for multisensor QPE for all of Colorado.

    gage_location : pandas Dataframe
        Dataframe with gage coordinates

    closest : pandas Dataframe
        Dataframe with closest 9 coordinates of MRMS 

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : animation
        Animation of 1hr QPE for each storm with gage locations.
    """
    # animation
    # things to add: topography, wind, pizza slices

    mrms = mrms.unknown.where(mrms.unknown>0)

    lon, lat = np.meshgrid(mrms.longitude,mrms.latitude)

    fig = plt.figure(1, figsize=(20,16))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                           hspace=0.01, wspace=0.01)

    plotcrs = ccrs.LambertConformal(central_latitude=38.5, central_longitude=-105)
    ax = plt.subplot(1,1,1, projection=plotcrs)
    ax.set_extent((252.3, 253.5, 39.2, 40)) # slightly larger area

    x = gage_location['Long']
    y = gage_location['Lat']
    ax.scatter(x,y,s=60,c='r',transform=ccrs.PlateCarree())
    closest_x=[]
    closest_y=[]
    for i in range(6):
        for j in np.arange(1,10,1):
            closest_x.append(closest.iloc[i][j][0])
            closest_y.append(closest.iloc[i][j][1])
    ax.scatter(closest_x,closest_y,s=60,c='b',transform=ccrs.PlateCarree())

    for i in gage_location.index:
        ax.text(x[i],y[i],gage_location['ID'][i],transform=ccrs.PlateCarree(), fontsize=20,color='r')

    plt.rcParams['animation.html']='jshtml'
    artists = []
    ds = []
    for j in storm_time.index:
        start = storm_time['start'][j]
        end = storm_time['end'][j]
        mrms_storm = mrms.sel(time=slice(start,end))    

        for i in range(len(mrms_storm)):
            ds = mrms_storm[i]
            this_time = 'Storm_ID: '+str(storm_time.iloc[j]['storm_id'])+', Storm Date: '+str(mrms_storm[i].time.values)[0:10]+', Hour: '+str(mrms_storm[i].time.values)[11:16]
            text = ax.text(0.5,1,this_time,ha='center',verticalalignment='bottom',transform=ax.transAxes, fontsize=20)
            mesh = ax.pcolormesh(lon,lat,ds,transform=ccrs.PlateCarree(),cmap='viridis', alpha=0.6,vmin=0,vmax=24)    
            artists.append([mesh,text])

    anim = ArtistAnimation(fig, artists, interval = 200)
    return anim

def animation_CO(mrms,gage_location,closest,storm_time):
    """The "animation_CO" function creates an animation of the 1hr QPE for all of CO.
    
    Parameters
    ----------
    mrms : xarray Dataset
        DataArray for multisensor QPE for all of Colorado.

    gage_location : pandas Dataframe
        Dataframe with gage coordinates

    closest : pandas Dataframe
        Dataframe with closest 9 coordinates of MRMS 

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : animation
        Animation of 1hr QPE for each storm with gage locations.
    """
    # animation
    # things to add: topography, wind, pizza slices

    mrms = mrms.unknown.where(mrms.unknown>0)

    lon, lat = np.meshgrid(mrms.longitude,mrms.latitude)

    fig = plt.figure(1, figsize=(20,16))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                           hspace=0.01, wspace=0.01)

    plotcrs = ccrs.LambertConformal(central_latitude=38.5, central_longitude=-105)
    ax = plt.subplot(1,1,1, projection=plotcrs)

    x = gage_location['Long']
    y = gage_location['Lat']
    ax.scatter(x,y,s=60,c='r',transform=ccrs.PlateCarree())
    closest_x=[]
    closest_y=[]
    for i in range(6):
        for j in np.arange(1,10,1):
            closest_x.append(closest.iloc[i][j][0])
            closest_y.append(closest.iloc[i][j][1])
    ax.scatter(closest_x,closest_y,s=60,c='b',transform=ccrs.PlateCarree())

    for i in gage_location.index:
        ax.text(x[i],y[i],gage_location['ID'][i],transform=ccrs.PlateCarree(), fontsize=20,color='r')

    plt.rcParams['animation.html']='jshtml'
    artists = []
    ds = []
    for j in storm_time.index:
        start = storm_time['start'][j]
        end = storm_time['end'][j]
        mrms_storm = mrms.sel(time=slice(start,end))    

        for i in range(len(mrms_storm)):
            ds = mrms_storm[i]
            this_time = 'Storm_ID: '+str(storm_time.iloc[j]['storm_id'])+', Storm Date: '+str(mrms_storm[i].time.values)[0:10]+', Hour: '+str(mrms_storm[i].time.values)[11:16]
            text = ax.text(0.5,1,this_time,ha='center',verticalalignment='bottom',transform=ax.transAxes, fontsize=20)
            mesh = ax.pcolormesh(lon,lat,ds,transform=ccrs.PlateCarree(),cmap='viridis', alpha=0.6,vmin=0,vmax=24)    
            artists.append([mesh,text])

    anim = ArtistAnimation(fig, artists, interval = 200)
    return anim


def correctedVuncorrected(mrms,mrms_uncorrected,gage,best_cell,title, gage_id, storm_time):
    """The "correctedVuncorrected" function plots gage vs best MRMS for multisensor corrected and radar only precip rates.
    
    Parameters
    ----------
    mrms : xarray Dataset
        DataArray for intensity calculated from multisensor QPE.
        
    mrms_uncorrected : xarray Dataset
        DataArray for intensity calculated from radar only QPE.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.
        
    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    title : string
        Title for plot

    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : plot
        Plot of all gage and MRMS values for best, mean, median, and max of cells.
    """
    fig,ax = plt.subplots(figsize=(20, 10), facecolor='w', edgecolor='k')
    g = []
    c = []
    u = []

    for i in range(len(gage_id)):
        for j in storm_time.index:
            start = storm_time['start'][j]
            end = storm_time['end'][j]

            gage_storm = gage[start:end][gage_id[i]]
            cell = best_cell[start:end][gage_id[i]+'cell']

            mrms_C_storm = mrms[i].sel(time=slice(start,end))
            mrms_U_storm = mrms_uncorrected[i].sel(time=slice(start,end))
            C = []
            U  = []
            n=0
            for k in cell:
                C.append(mrms_C_storm[n,k,k].values)
                U.append(mrms_U_storm[n,k,k].values)
                n+=1
            g.append(gage_storm.values)
            u.append(U)
            c.append(C)

            plt.scatter(gage_storm,C,c='r',marker='^')
            plt.scatter(gage_storm,U,c='b',marker='x')

            x1 = (0,np.max(gage_storm))
            y1 = (0,np.max(gage_storm))
            plt.plot(x1,y1,'g')

            plt.ylabel('MRMS (mm/hr)',size=14)
            plt.xlabel('gage (mm/hr)',size=14)
            plt.legend(['corrected','uncorrected'])

    g = np.concatenate(g)
    u = np.concatenate(u)
    c = np.concatenate(c)

    rmse_corrected = mean_squared_error(g, c, squared=False)
    rmse_uncorrected = mean_squared_error(g, u, squared=False)
    
    textstr = '\n'.join((
        r'$rmse corrected=%.2f$' % (rmse_corrected, ),
        r'$rmse uncorrected=%.2f$' % (rmse_uncorrected, )))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.9, 0.8, textstr, transform=ax.transAxes,fontsize=14,verticalalignment='top',bbox=props)
    plt.title(title,y=.95,size=20)

    return

def stormlength_bias(bias, title, gage_id, storm_time):
    """The "stormlength_bias" function plots the storm duration vs bias.

    Parameters
    ----------
    bias : pandas DataFrame
        Either total accumulation bias or max intensity bias

    title : string
        Title for plot

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.

    Returns
    ------
    output : plot
        Plot of storm duration vs bias at all gages.
    """

    fig = plt.subplots(figsize=(20, 10), facecolor='w', edgecolor='k')
    bias['bias_opt'] = np.abs(1-bias['bias'])
    plt.xlabel('Storm Length',size=14)
    plt.ylabel('max MRMS/max gage',size=14)
    plt.title(title,y=0.95,size=20)
    n = 6
    color = cm.rainbow(np.linspace(0, 1, n))
    #plt.xticks(np.arange(storm_time['storm_id'].min(), storm_time['storm_id'].max()+1, 1.0))

    for i in storm_time.index:
        for j in range(len(gage_id)):
            c = color[j]
            best_bias = bias.loc[(bias['storm_id']==i)&(bias['gage_id']==gage_id[j])]['bias_opt'].min()
            plot_bias = bias.loc[(bias['storm_id']==i)&(bias['gage_id']==gage_id[j])&(bias['bias_opt']==best_bias)].min()['bias']
            plt.scatter(storm_time.iloc[i]['length'].seconds/60**2,plot_bias,color=c)
    plt.legend(['GCFC2', 'GCEC2', 'GCNC2', 'GCTC2', 'GCCC2', 'GCDC2'])

    return

def gage_bias(bias, gage,title, xaxis,gage_id, storm_time):
    """The "bias_storm" function plots the total accumulation bias or the maximum intensity bias for each storm.

    Parameters
    ----------
    bias : pandas DataFrame
        Either total accumulation bias or max intensity bias
        
    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.

    title : string
        Title for plot

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.

    Returns
    ------
    output : plot
        Plot of storm vs bias at all gages.
    """

    fig = plt.subplots(figsize=(20, 10), facecolor='w', edgecolor='k')
    bias['bias_opt'] = np.abs(1-bias['bias'])
    plt.xlabel(xaxis,size=14)
    plt.ylabel('max gage/max MRMS',size=14)
    plt.title(title,y=0.95,size=20)
    n = 6
    color = cm.rainbow(np.linspace(0, 1, n))
    #plt.xticks(np.arange(storm_time['storm_id'].min(), storm_time['storm_id'].max()+1, 1.0))

    for i in storm_time.index:
        for j in range(len(gage_id)):
            start = storm_time['start'][i]
            end = storm_time['end'][i]
            gage_storm = gage[start:end][gage_id[j]]
            
            c = color[j]
            best_bias = bias.loc[(bias['storm_id']==i)&(bias['gage_id']==gage_id[j])]['bias_opt'].min()
            plot_bias = bias.loc[(bias['storm_id']==i)&(bias['gage_id']==gage_id[j])&(bias['bias_opt']==best_bias)].min()['bias']
            plt.scatter(gage_storm.max(),plot_bias,color=c)
    plt.legend(['GCFC2', 'GCEC2', 'GCNC2', 'GCTC2', 'GCCC2', 'GCDC2'])

    return

def best_gageverror(gage,mrms,best_cell,gage_id,storm_time):
    """The "best_gageverror" function plots gage magnitude vs the error between the best coordinate and the gage.
    
    Parameters
    ----------
    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.    

    mrms : xarray Dataset
        DataArray for intensity calculated from multisensor QPE at the gage.
        
    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : plot
        Plot of gage intensity vs error.
    """
    fig, axs = plt.subplots(figsize=(20, 10), facecolor='w', edgecolor='k')
    fig.suptitle('15-min intensity, Gage v Residuals',y=0.95,size=20)

    for i in range(len(gage_id)):
        for j in storm_time.index:
            start = storm_time['start'][j]
            end = storm_time['end'][j]

            gage_storm = gage[start:end][gage_id[i]]
            
            cell = best_cell[start:end][gage_id[i]+'cell']
            
            mrms_C_storm = mrms[i].sel(time=slice(start,end))
            
            C = []
            n=0
            for k in cell:
                C.append(mrms_C_storm[n,k,k].values)
                n+=1
                
            error = gage_storm - C

            plt.scatter(gage_storm,error,c='r',marker='^')         
            plt.ylabel('Gage - MRMS(mm/hr)',size=14)
            plt.xlabel('gage (mm/hr)',size=14)
  
    return 

def boxplots(gage,mrms,best_cell, title,gage_id,storm_time):
    """The "boxplots" function creates a boxplot for each gage showing the spread of error for each storm.
    
    Parameters
    ----------
    mrms : xarray Dataset
        DataArray for intensity calculated from multisensor QPE.

    gage : pandas Dataframe
        Pandas dataframe with with the precipitation rate from the gage.
        
    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    title : string
        Title for plot.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.
    
    Returns
    ------
    output : subplot
        Subplot with boxplots for each gage.
    """
    fig, axs = plt.subplots(2,3, figsize=(20, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .2, wspace=.1)
    fig.suptitle(title,size=20)
    fig.text(0.5, 0.04, 'storm',ha='center',size=20)
    fig.text(0.08, 0.5, 'Gage - MRMS(mm/hr)', va='center', rotation='vertical',size=20)

    axs = axs.ravel()
    count = 0
    for i in range(len(gage_id)):
        error = []
        for j in storm_time.index:
            start = storm_time['start'][j]
            end = storm_time['end'][j]
            gage_storm = gage[start:end][gage_id[i]]
            cell = best_cell[start:end][gage_id[i]+'cell']
            mrms_C_storm = mrms[i].sel(time=slice(start,end))
            
            C = []
            n=0
            for k in cell:
                C.append(mrms_C_storm[n,k,k])
                n+=1
                
            error.append(gage_storm - C)
            
        axs[count].set_title('Gage_ID:'+gage_id[i],fontsize=14)
        axs[count].boxplot(error)
        #axs[count].set_ylim(0,2)
        count+=1
    
    return

def bestcell_bygage_bias(bias,gage_id,storm_time):
    """The "bestcell_bygage_bias" function creates a heatmap for each gage showing the bias for each 
    coordinate around the gage.
    
    Parameters
    ----------
    bias : pandas DataFrame
        Either total accumulation bias or max intensity bias

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.

    Returns
    ------
    output : heatmap
        Heatmap for each gage.
    """
    # suppress false pos warning
    pd.options.mode.chained_assignment = None
    
    # per gage
    best_cell = []

    for i in gage_id:
        for j in storm_time.index:
            df = bias.loc[(bias['gage_id']==i) & (bias['storm_id']==j)]
            df['bias_opt']= np.abs(1-df['bias'])
            
            best_cell.append(df[df['bias_opt']==df['bias_opt'].min()])

    best_cell=pd.concat(best_cell)

    # find which cell has lowest bias and create new dataframe
    best_bygage = []

    for i in gage_id:
        df = best_cell.loc[best_cell['gage_id']==i]
        best = df['cell_id'].values
        unique, counts = np.unique(best, return_counts=True)
        prob = counts/(counts.sum())
        best_bygage.append([unique,prob,i])
        
    plot_gage = []
    for i in range(len(best_bygage)):
        df = pd.DataFrame(best_bygage[i][1],index=best_bygage[i][0],columns=[best_bygage[i][2]])
        plot_gage.append(df)

    plot_gage = pd.concat(plot_gage,axis=1).sort_index()
    
    fig, axs = plt.subplots(2,3, figsize=(15, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 0.1, wspace=0)
    fig.suptitle('Probability of bias being closest to 1, ',size=20)
    axs = axs.ravel()
    
    k = 0
    for i in gage_id:
        df = plot_gage[i]
        probability = [df.iloc[[6,1,5]],df.iloc[[4,0,3]],df.iloc[[8,2,7]]]

        im = axs[k].imshow(probability, cmap = 'Greens')
        plt.colorbar(im,ax=axs[k])
        axs[k].set_xticks([])
        axs[k].set_yticks([])
        axs[k].set_title ('Gage id: '+i)
        k+=1

    return

def bestcell_bygage(best_cell,gage_id,storm_time,closest):
    """The "bestcell_bygage" function creates a heatmap for each gage showing the intensity 
    error for each coordinate around the gage.
    
    Parameters
    ----------
    best_cell : pandas DataFrame
        A pandas DataFrame identifying the coordinate with the lowest error.

    gage_id : Pandas DataFrame
        A pandas DataFrame with the ID's of gages

    storm_time : Pandas DataFrame
        Pandas dataframe with the storm times calculated from the gage accumulation.

    closest : pandas Dataframe
        Outputs a dataframe with the closest coordinate and nearest 8 coordinates for each gage

    Returns
    ------
    output : heatmap
        Heatmap for each gage.
    """
        
    prob_storm = []
    index = []
    cells = closest.columns[1:10].values
    for i in range(len(gage_id)):
        for j in storm_time.index:
            start = storm_time['start'][j]
            end = storm_time['end'][j]
            best_storm = best_cell[start:end]
            best = best_storm.loc[best_storm[gage_id[i]+'dupicated']==False]
            best_c = best[gage_id[i]+'cell'].values

            n,bins = np.histogram(best_c,bins=np.arange(0,10,1))
            probability = n/len(best_c)

            prob_storm.append([[j]*9,[gage_id[i]]*9,cells,probability,n])

    prob_storm = [pd.DataFrame(prob_storm[i]).T for i in range(len(prob_storm))]
    prob_storm = pd.concat(prob_storm)
    prob_storm = prob_storm.rename(columns={0:'storm_id',1:'gage_id',2:'cell_id',3:'probability',4:'count'})
    
    fig, axs = plt.subplots(2,3, figsize=(15, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 0.1, wspace=0)
    fig.suptitle('Probability of lowest error, ',size=20)
    axs = axs.ravel()
    k=0
    for j in gage_id:

        df = prob_storm.loc[prob_storm['gage_id']==j]['probability'].dropna()
        probability = [df.iloc[[6,1,5]].values.tolist(),df.iloc[[4,0,3]].values.tolist(),df.iloc[[8,2,7]].values.tolist()]

        im = axs[k].imshow(probability, cmap = 'Greens')
        axs[k].set_xticks([])
        axs[k].set_yticks([])
        axs[k].set_title ('Gage id: '+j)
        k+=1
    return

def plot_RQI(codtm,RQI):
    colo_aoi = np.array([-109.04667027,   37.00084591, -102.01278328,   41.00085827])

    # Get lat min, max
    aoi_lat = [float(colo_aoi[1]), float(colo_aoi[3])]
    aoi_lon = [float(colo_aoi[0]), float(colo_aoi[2])]
    # Notice that the longitude values have negative numbers
    # we need these values in a global crs so we can subtract from 360
    # The mrms files use a global lat/lon so adjust values accordingly
    aoi_lon[0] = aoi_lon[0] + 360
    aoi_lon[1] = aoi_lon[1] + 360

    #   DID I SCREW THIS UP?!
    RQI_CO = RQI.sel(longitude=slice(aoi_lon[0], aoi_lon[1]),latitude=slice(aoi_lat[0], aoi_lat[1]))

        # find dtm values of x closest to mrms coords
    xelev = codtm.x
    xelev = xelev.to_numpy()

    xprecip = RQI_CO.longitude
    xprecip = xprecip.to_numpy()

    index=[]
    for i in range(len(xprecip)):
        diff = np.absolute(xelev-xprecip[i])
        index.append(diff.argmin())
        
    xmatch = [xelev[i] for i in index]

    # find dtm values of y closest to mrms coords
    yelev = codtm.y
    yelev = yelev.to_numpy()

    yprecip = RQI_CO.latitude
    yprecip = yprecip.to_numpy()

    index=[]
    for i in range(len(yprecip)):
        diff = np.absolute(yelev-yprecip[i])
        index.append(diff.argmin())
        
    ymatch = [yelev[i] for i in index]

    # slice elevation based on xmatch and ymatch
    newelev = codtm.sel(x=xmatch,y=ymatch)

    # create another xarray where lon lat exact match with precip
    # AND drop extra band dim in codtm
    newelev = newelev.assign_coords(x=xprecip)
    newelev = newelev.assign_coords(y=yprecip)

    newelev = newelev.rename({'x': 'longitude','y': 'latitude'})

    newelev = newelev.drop_vars('band')
    noband = newelev.sel(band=0)

    RQI = RQI_CO.where(RQI_CO>=0)
    lon, lat = np.meshgrid(noband.longitude,noband.latitude)

    # create mesh of all data
    fig = plt.figure(1, figsize=(20,16))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                        hspace=0.01, wspace=0.01)

    plotcrs = ccrs.LambertConformal(central_latitude=38.5, central_longitude=-105)
    ax = plt.subplot(gs[0], projection=plotcrs)

    ax.set_extent((-109.04667027, -102.01278328,  37.00084591,    41.00085827))

    ax.add_feature(cfeature.STATES, linewidth=1)

    elev=ax.contourf(lon,lat, noband.values, levels=list(range(2000, 5000, 500)), origin='upper',cmap='terrain', 
                alpha=0.8,transform=ccrs.PlateCarree())

    mesh = ax.pcolormesh(lon,lat,RQI.unknown,transform=ccrs.PlateCarree(),cmap='gray', alpha=0.4,vmin=0,vmax=1)  
    cb =fig.colorbar(elev,orientation="horizontal", shrink=.6,pad=0)
    cb.ax.tick_params(labelsize=14)
    cb.set_label("elevation (m)", fontsize=14)
    cb2 =fig.colorbar(mesh,orientation="horizontal",ax=ax, shrink=.6, pad=0.02)
    cb2.ax.tick_params(labelsize=20)
    cb2.set_label("Radar Quality Index", fontsize=20)

    return