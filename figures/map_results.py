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

state = pd.read_feather('../output/stateclean')
state_results = pd.read_feather('../output/state_results')
state_results=state_results.divide(state.max_mrms.values,axis=0)
state['qgb_t 0.05'] = state_results['qgb_t 0.05'].values
state['qgb_t 0.50'] = state_results['qgb_t 0.50'].values
state['qgb_t 0.95'] = state_results['qgb_t 0.95'].values


#%%
# look at median
med = state.groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.50']
med = med.to_xarray()
med_std = state.groupby(['mrms_lat','mrms_lon']).quantile(.25)['qgb_t 0.50']
med_std = med_std.to_xarray()

low = state.groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.05']
low = low.to_xarray()

high = state.groupby(['mrms_lat','mrms_lon']).median()['qgb_t 0.95']
high = high.to_xarray()
#%%
# open window values and calculate proportion of samples from each coord
df = pd.read_feather('../output/window_values_new')
df = df.loc[df.total_mrms_accum>0].reset_index(drop=True)
df = df.dropna()
#test = df.loc[df.norm_diff>70]
#test.groupby(['start','mrms_lat','mrms_lon']).count()
# this gage has most big outliers, see code above, remove it
df = df.loc[(df.mrms_lat!=40.57499999999929)&(df.mrms_lon!=254.91499899999639)]

df = df.loc[(df.total_mrms_accum>1)].reset_index(drop=True)
count = df.groupby(['mrms_lat','mrms_lon']).count().accum/len(df)
count = count.reset_index()
#%%
#tell plotter what to plot
plot_map = med
name_cb = 'median of nRMSE prediction from '+r'$\alpha$ = 0.50'+' model'
y,x = med.mrms_lat,med.mrms_lon

plot_map2 = med_std
name_cb2 = 'first quartile of nRMSE prediction from '+r'$\alpha$ = 0.50'+' model'
levels = list(np.arange(0.3,0.8,.1))

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
#%%
# map of gage
plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)

fig,axs = plt.subplots(1,2, subplot_kw=dict(projection=plotcrs), figsize=(15*.6,8*.6))


gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                       hspace=0.01, wspace=0.01)

# Set plot bounds -- or just comment this out if wanting to plot the full domain
axs[0].set_extent((-109.2, -103.5, 36.8, 41.3))
axs[0].set_title('(a)')
elev=axs[0].contourf(x,y,plot_map, cmap=cmap2,origin='upper', transform=ccrs.PlateCarree(),extend='both',levels=levels)


cb =plt.colorbar(elev, orientation="horizontal", shrink=.5,pad=0.01,ax=axs[0])
cb.ax.tick_params(labelsize=8)
cb.set_label(name_cb, fontsize=10)

axs[0].scatter(cities_df.longitude,cities_df.latitude,
           transform=ccrs.PlateCarree(),s = 25, facecolors='red',edgecolors='black',marker='^',label='city')

axs[0].scatter(radar_df.longitude,radar_df.latitude,
           transform=ccrs.PlateCarree(),s = 25, edgecolors='black',facecolors='#65FE08',marker='o',label='radar')
cmap_data = [
(255,255,212),
(254,217,142),
(254,153,41),
(217,95,14),
(153,52,4),
]
cmap_data_N=[]
for i in cmap_data:
    cmap_data_N.append([c/255 for c in i])

cmap = ListedColormap(cmap_data_N)
# define the bins and normalize
bounds = np.arange(0, .015, .0025)

norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

g=axs[0].scatter(count.mrms_lon,count.mrms_lat,
           transform=ccrs.PlateCarree(),s = 3, c = count.accum.values,cmap=cmap,norm=norm)

axs[0].add_feature(cfeature.STATES, linewidth=1)

axs[0].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')

gl = axs[0].gridlines(crs=ccrs.PlateCarree(), 
                  alpha=0, 
                  draw_labels=True, 
                  dms=True, 
                  x_inline=False, 
                  y_inline=False)
gl.xlabel_style = {'rotation':0}
# add these before plotting
gl.bottom_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels
legend = axs[0].legend()
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')

###############second plot
#################################
axs[1].set_extent((-109.2, -103.5, 36.8, 41.3))
axs[1].set_title('(b)')
elev=axs[1].contourf(x,y,plot_map2, cmap=cmap2,origin='upper', transform=ccrs.PlateCarree(),extend='both',levels=levels)


cb =plt.colorbar(elev, orientation="horizontal", shrink=.5,pad=0.01,ax=axs[1])
cb.ax.tick_params(labelsize=8)
cb.set_label(name_cb2, fontsize=10)

axs[1].scatter(cities_df.longitude,cities_df.latitude,
           transform=ccrs.PlateCarree(),s = 25, facecolors='red',edgecolors='black',marker='^',label='city')

axs[1].scatter(radar_df.longitude,radar_df.latitude,
           transform=ccrs.PlateCarree(),s = 25, edgecolors='black',facecolors='#65FE08',marker='o',label='radar')

g=axs[1].scatter(count.mrms_lon,count.mrms_lat,
           transform=ccrs.PlateCarree(),s = 3, c = count.accum.values,cmap=cmap,norm=norm)

cbar2 = plt.colorbar(g, orientation='vertical', shrink=.55,pad=0.01,ax=axs[1])
cbar2.ax.tick_params(labelsize=8)
cbar2.set_label("proportion of samples", fontsize=10)

axs[1].add_feature(cfeature.STATES, linewidth=1)

axs[1].add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')

gl = axs[1].gridlines(crs=ccrs.PlateCarree(), 
                  alpha=0, 
                  draw_labels=True, 
                  dms=True, 
                  x_inline=False, 
                  y_inline=False)
gl.xlabel_style = {'rotation':0}
# add these before plotting
gl.bottom_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels
gl.left_labels=False
legend = axs[1].legend()
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
plt.subplots_adjust(wspace=0.001)
plt.tight_layout()
#%%

fig.savefig('../output_figures/resultmap.pdf',
           bbox_inches='tight',dpi=255,transparent=False,facecolor='white')

# %%