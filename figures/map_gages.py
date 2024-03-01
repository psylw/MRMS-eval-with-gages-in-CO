#%%

import rioxarray as rxr
import matplotlib.gridspec as gridspec
from metpy.plots import USCOUNTIES
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# open window values and calculate proportion of samples from each coord
df = pd.read_feather('../output/window_values_new')
df = df.loc[df.total_mrms_accum>0].reset_index(drop=True)
df = df.dropna()
#test = df.loc[df.norm_diff>70]
#test.groupby(['start','mrms_lat','mrms_lon']).count()
# this gage has most big outliers, see code above, remove it
df = df.loc[(df.mrms_lat!=40.57499999999929)&(df.mrms_lon!=254.91499899999639)]

df = df.loc[(df.total_mrms_accum>1)].reset_index(drop=True)

source = df.groupby(['gage_lat','gage_lon']).agg(list).gage_source.reset_index()
source['gage_source'] = [np.unique(source.iloc[i].gage_source)[0] for i in source.index]

data_folder = os.path.join('..','..','data','precip_gage')
# add mesowest
meso_gages = pd.read_csv(data_folder+'\mesowest.csv')

source.loc[(source.gage_lat.isin(meso_gages.lat)) & (source.gage_lon.isin(meso_gages.lon + 360)), 'gage_source'] = 'mesowest'
#%%
# Updated list of major cities in Colorado with their coordinates
cities = {'City': ['Denver', 'Colorado Springs', 'Fort Collins', 'Grand Junction', 'pueblo'],
          'latitude': [39.7392, 38.8339, 40.5853, 39.0639,38.2544],
          'longitude': [-104.9903, -104.8214, -105.0844, -108.5506,-104.6091]}

cities_df = pd.DataFrame(cities)

radar = {'name':['kcys','kftg','kpux','kala','kgjx'],
         'latitude':[41.17,39.8, 38.47,37.46,39.05],
         'longitude':[-104.82, -104.56,-104.2,-105.86,-108.23]}
radar_df = pd.DataFrame(radar)
#%%
datafile1 = "../../../data/elev_data/CO_SRTM1arcsec__merge.tif"
codtm = rxr.open_rasterio(datafile1)
newelev = codtm.drop_vars('band')

noband = newelev.sel(band=0)

noband = noband[{'x': slice(None, None, 10), 'y': slice(None, None, 10)}][{}]
lon, lat = np.meshgrid(noband.x,noband.y)


fig = plt.figure(1, figsize=(14*.9,10*.9))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, .02], bottom=.07, top=.99,
                       hspace=0.01, wspace=0.01)

plotcrs = ccrs.LambertConformal(central_latitude=(41.3+36.8)/2, central_longitude=(-109.2-103.5)/2)
ax = plt.subplot(1,1,1, projection=plotcrs)
ax.set_extent((-109.2, -103.5, 36.8, 41.3))

elev=ax.contourf(lon,lat, noband, levels=list(range(2000, 5000, 500)), origin='upper',cmap='terrain', 
            alpha=0.4,transform=ccrs.PlateCarree())
cb =fig.colorbar(elev,orientation="horizontal", shrink=.55,pad=0.01)
cb.ax.tick_params(labelsize=12)
cb.set_label("elevation (m)", fontsize=12)

ax.add_feature(cfeature.STATES, linewidth=1)

ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')
ax.scatter(cities_df.longitude,cities_df.latitude,
           transform=ccrs.PlateCarree(),s = 120, facecolors='red',marker="^",label='city',edgecolors='black')

ax.scatter(radar_df.longitude,radar_df.latitude,
           transform=ccrs.PlateCarree(),s =150,facecolors='cornflowerblue',marker='o',label='radar',edgecolors='black')
gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                  alpha=0, 
                  draw_labels=True, 
                  dms=True, 
                  x_inline=False, 
                  y_inline=False)
gl.xlabel_style = {'rotation':0, 'fontsize':12}
gl.ylabel_style = {'fontsize':12}
# add these before plotting
gl.bottom_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels

df1 = source.loc[source.gage_source=='usgs']
df2 = source.loc[source.gage_source=='csu']
df3 = source.loc[source.gage_source=='coagmet']
df4 = source.loc[source.gage_source=='mesowest']
plt.scatter(df1.gage_lon,df1.gage_lat,transform=ccrs.PlateCarree(),label='USGS',marker='x',color='darkred',s =15,linewidth=1)
plt.scatter(df2.gage_lon,df2.gage_lat,transform=ccrs.PlateCarree(),label='CSU',marker='+',color='fuchsia',s =15,linewidth=1)
plt.scatter(df3.gage_lon,df3.gage_lat,transform=ccrs.PlateCarree(),label='CoAgMET',marker='*',color='blue',s =15,linewidth=1)
plt.scatter(df4.gage_lon,df4.gage_lat,transform=ccrs.PlateCarree(),label='MesoWest',marker='.',color='orange',s =20,linewidth=2)
#plt.scatter(df4.longitude,df4.latitude,transform=ccrs.PlateCarree(),label='ARS',marker='x',color='purple')

legend = ax.legend(fontsize=12,loc='upper left')
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')


#plt.legend()

fig.savefig("../output_figures/f01.pdf",
           bbox_inches='tight',dpi=600,transparent=False,facecolor='white')
# %%
