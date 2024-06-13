
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

state = pd.read_feather('../output/experiments/stateclean_year')
state_results = pd.read_feather('../output/state_results')

input_state_results=state_results.divide(state.max_mrms.values,axis=0)
levels = list(np.arange(0.3,0.8,.1))
label1 = 'median of nRMSE prediction'

state_input = pd.concat([state.reset_index(drop=True),input_state_results],axis=1)

month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
#%%
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

fig,axs = plt.subplots(1,5, subplot_kw=dict(projection=plotcrs), figsize=(18,8))

for idx,month in enumerate(state_input.month.unique()):
    print(month_names[month - 1])
    state = state_input[state_input.month==month]
    # look at median
    med = state.groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.50']
    med = med.to_xarray()

    #tell plotter what to plot
    plot_map = med
    name_cb = label1
    y,x = med.mrms_lat,med.mrms_lon

    # Set plot bounds -- or just comment this out if wanting to plot the full domain
    axs[idx].set_extent((-109.2, -103.5, 36.8, 41.3))

    elev=axs[idx].contourf(x,y,plot_map, cmap=cmap2,origin='upper', transform=ccrs.PlateCarree(),extend='both',levels=levels)

    axs[idx].scatter(cities_df.longitude,cities_df.latitude,
            transform=ccrs.PlateCarree(),s = 25, facecolors='red',edgecolors='black',marker='^',label='city')

    axs[idx].scatter(radar_df.longitude,radar_df.latitude,
            transform=ccrs.PlateCarree(),s = 25, edgecolors='black',facecolors='cornflowerblue',marker='o',label='radar')

    axs[idx].add_feature(cfeature.STATES, linewidth=1)

    axs[idx].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')
    if idx==0:
        gl = axs[idx].gridlines(crs=ccrs.PlateCarree(), 
                        alpha=0, 
                        draw_labels=True, 
                        dms=True, 
                        x_inline=False, 
                        y_inline=False)
        gl.xlabel_style = {'rotation':0}
        # add these before plotting
        gl.bottom_labels=False   # suppress top labels
        gl.right_labels=False # suppress right labels
    else:
        gl = axs[idx].gridlines(crs=ccrs.PlateCarree(), 
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
    legend = axs[idx].legend()
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('white')
    axs[idx].set_title(month_names[month - 1], fontsize=14)
plt.tight_layout()
cb =plt.colorbar(elev, shrink=.4,pad=0.01,ax=axs)
cb.ax.tick_params(labelsize=12)
cb.set_label(name_cb, fontsize=14)
fig.savefig("../output_figures/experiments/S19.pdf",bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
