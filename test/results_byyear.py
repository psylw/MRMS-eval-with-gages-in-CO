
# plot results for each year for different approaches to separate changes in MRMS

# %%
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import matplotlib.gridspec as gridspec
from metpy.plots import USCOUNTIES
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

state = pd.read_feather('../output/experiments/stateclean_year')

state_results_original = pd.read_feather('../output/state_results')
state_results_pre = pd.read_feather('../output/experiments/state_results_pre_2021')
state_results_post = pd.read_feather('../output/experiments/state_results_post_2021')

input_state_results1 = state_results_original.divide(state.max_mrms.values,axis=0)
input_state_results2 = state_results_pre.divide(state[state.year<2021].max_mrms.values,axis=0)
input_state_results3 = state_results_post.divide(state[state.year>= 2021].max_mrms.values,axis=0)

state_input1 = pd.concat([state.reset_index(drop=True),input_state_results1],axis=1)
state_input2 = pd.concat([state[state.year<2021].reset_index(drop=True),input_state_results2],axis=1)
state_input3 = pd.concat([state[state.year>= 2021].reset_index(drop=True),input_state_results3],axis=1)


#%%
cmap_data = [
(255,255,217),
(237,248,177),
(199,233,180),
(127,205,187),
(65,182,196),
(29,145,192),
(34,94,168),
(37,52,148),
(8,29,88),]

cmap_data_N=[]
for i in cmap_data:
    cmap_data_N.append([c/255 for c in i])

cmap2 = LinearSegmentedColormap.from_list('custom',cmap_data_N)
cmap1 = LinearSegmentedColormap.from_list('custom',cmap_data_N[::-1])

# Updated list of major cities in Colorado with their coordinates
cities = {'City': ['Denver', 'Colorado Springs', 'Fort Collins', 'Grand Junction', 'pueblo'],
        'latitude': [39.7392, 38.8339, 40.5853, 39.0639,38.2544],
        'longitude': [-104.9903, -104.8214, -105.0844, -108.5506,-104.6091]}

cities_df = pd.DataFrame(cities)

radar = {'name':['kcys','kftg','kpux','kala','kgjx'],
        'latitude':[41.17,39.8, 38.47,37.46,39.05],
        'longitude':[-104.82, -104.56,-104.2,-105.86,-108.23]}
radar_df = pd.DataFrame(radar)

plt.rcParams['figure.dpi'] = 150

# map of gage

plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)



gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                    hspace=0.01, wspace=0.01)

fig,axs = plt.subplots(2,3, subplot_kw=dict(projection=plotcrs), figsize=(12*.8,10*.75))

for idx in range(6):
    if idx ==0:
        med = state_input1[state_input1.year<2021].groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.50']
        cmap = cmap2
        title = 'a) original model pre v12.0'
        levels = list(np.arange(0.3,0.8,.1))
        ax1 = 0
        ax2 = 0
    elif idx ==1:
        med = state_input1[state_input1.year>=2021].groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.50']
        cmap = cmap2
        title = 'b) original model post v12.0'
        levels = list(np.arange(0.3,0.8,.1))
        ax1 = 1
        ax2 = 0
        name_cb = 'median of nRMSE prediction'
    elif idx ==2:
        med = state_input2.groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.50']
        cmap = cmap2
        title = 'c) model pre v12.0'
        levels = list(np.arange(0.3,0.8,.1))
        ax1 = 0
        ax2 = 1
    elif idx ==3:
        med = state_input3.groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.50']
        cmap = cmap2
        title = 'd) model post v12.0'
        levels = list(np.arange(0.3,0.8,.1))
        ax1 = 1
        ax2 = 1
        name_cb = 'median of nRMSE prediction'
    elif idx ==4:
        med = state_input1[state_input1.year<2021].groupby(['mrms_lat','mrms_lon']).median()['rqi_min']
        cmap = cmap1
        title = 'e) RQI pre v12.0'
        levels = list(np.arange(.15,.95,.15))
        ax1 = 0
        ax2 = 2
    else:
        med = state_input1[state_input1.year>=2021].groupby(['mrms_lat','mrms_lon']).median()['rqi_min']
        cmap = cmap1
        title = 'f) RQI post v12.0'
        levels = list(np.arange(.15,.95,.15))
        ax1 = 1
        ax2 = 2
        name_cb = 'median of RQI min'
    med = med.to_xarray()

    # Set plot bounds -- or just comment this out if wanting to plot the full domain
    axs[ax1,ax2].set_extent((-109.2, -103.5, 36.8, 41.3))

    elev=axs[ax1,ax2].contourf(med.mrms_lon,med.mrms_lat,med, cmap=cmap,origin='upper', transform=ccrs.PlateCarree(),extend='both',levels=levels)

    axs[ax1,ax2].scatter(cities_df.longitude,cities_df.latitude,
            transform=ccrs.PlateCarree(),s = 25, facecolors='red',edgecolors='black',marker='^',label='city')

    axs[ax1,ax2].scatter(radar_df.longitude,radar_df.latitude,
            transform=ccrs.PlateCarree(),s = 25, edgecolors='black',facecolors='cornflowerblue',marker='o',label='radar')

    axs[ax1,ax2].add_feature(cfeature.STATES, linewidth=1)

    axs[ax1,ax2].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')

    axs[ax1,ax2].set_title(title, fontsize=14)


    if ax1==0 and ax2==0:
        gl = axs[ax1,ax2].gridlines(crs=ccrs.PlateCarree(), 
                        alpha=0, 
                        draw_labels=True, 
                        dms=True, 
                        x_inline=False, 
                        y_inline=False)
        gl.xlabel_style = {'rotation':0}
        # add these before plotting
        gl.bottom_labels=False   # suppress top labels
        gl.right_labels=False # suppress right labels
    elif ax1==0:

        gl = axs[ax1,ax2].gridlines(crs=ccrs.PlateCarree(), 
                        alpha=0, 
                        draw_labels=True, 
                        dms=True, 
                        x_inline=False, 
                        y_inline=False)
        gl.xlabel_style = {'rotation':0}
        # add these before plotting
        gl.bottom_labels=False   # suppress top labels
        gl.right_labels=False # suppress right labels
        gl.left_labels=False # suppress right labels
    elif ax1==1 and ax2==0:
        cb =plt.colorbar(elev, orientation="horizontal", shrink=.8,pad=0.01,ax=axs[ax1,ax2])
        cb.ax.tick_params(labelsize=12)
        cb.set_label(name_cb, fontsize=14)
        gl = axs[ax1,ax2].gridlines(crs=ccrs.PlateCarree(), 
                        alpha=0, 
                        draw_labels=True, 
                        dms=True, 
                        x_inline=False, 
                        y_inline=False)
        gl.xlabel_style = {'rotation':0}
        # add these before plotting
        gl.bottom_labels=False   # suppress top labels
        gl.right_labels=False # suppress right labels
        gl.top_labels=False # suppress right labels
    elif ax1==1:
        cb =plt.colorbar(elev, orientation="horizontal", shrink=.8,pad=0.01,ax=axs[ax1,ax2])
        cb.ax.tick_params(labelsize=12)
        cb.set_label(name_cb, fontsize=14)
    else:
        gl = axs[ax1,ax2].gridlines(crs=ccrs.PlateCarree(), 
                        alpha=0, 
                        draw_labels=True, 
                        dms=True, 
                        x_inline=False, 
                        y_inline=False)
        gl.xlabel_style = {'rotation':0}
        # add these before plotting
        gl.bottom_labels=False   # suppress top labels
        gl.right_labels=False # suppress right labels
        gl.top_labels=False # suppress right labels
        gl.left_labels=False # suppress right labels

    legend = axs[ax1,ax2].legend()
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('white')

plt.tight_layout()

fig.savefig("../output_figures/experiments/f04.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
