import geopandas as gpd
import cartopy.crs as ccrs
import pandas as pd
import numpy as np


def remove_closest(df,distance_lessthan):
    locations = df.groupby(['latitude','longitude']).count().reset_index().iloc[:,0:2]
    x = locations.longitude.values
    y = locations.latitude.values

    points = gpd.GeoSeries.from_xy(x, y, crs="EPSG:4326")
    crs = ccrs.LambertConformal(central_latitude=38.5, central_longitude=-105)
    points = points.to_crs(crs)

    min_distance=[]
    for i in points.index:
        p = points[i]

        distance = []
        for j in points.index:
            distance.append(p.distance(points[j])/1000)

        distance = np.asarray(distance)
        idx = np.argwhere(distance<=distance_lessthan)
        min_distance.append({'coord_idx':i,'drop_idx':points.loc[points.index.isin(idx[:,0])].index.values})

    min_distance = pd.DataFrame(min_distance)

    for i in min_distance.index:
        drops = np.asarray(min_distance.drop_idx[i])
        drops = drops[np.where(drops>min_distance.coord_idx[i])]
        if len(drops)>0:
            min_distance.drop_idx[i] = drops
        else:
            min_distance.drop_idx[i] = np.nan

    drop = [i for i in min_distance.drop_idx.dropna()]
    drop = np.concatenate(drop)

    df['coord'] = list(zip(df.latitude,df.longitude))
    locations = locations.iloc[~locations.index.isin(drop)]

    locations = list(zip(locations.latitude,locations.longitude))
    df = df.loc[df.coord.isin(locations)].drop(columns='coord').reset_index(drop=True)
    return df
