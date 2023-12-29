
import rioxarray as rxr
import numpy as np

def get_elev(dem,data_folder,ds):

    # open geotiff and create xarray
    datafile1 = data_folder+dem
    codtm = rxr.open_rasterio(datafile1)
    # change lon to match global lat/lon in grib file
    codtm = codtm.assign_coords(x=(((codtm.x + 360))))

    # find dtm values of x closest to mrms coords
    xelev = codtm.x
    xelev = xelev.to_numpy()

    xprecip = ds.longitude
    xprecip = xprecip.to_numpy()

    index=[]
    for i in range(len(xprecip)):
        diff = np.absolute(xelev-xprecip[i])
        index.append(diff.argmin())

    xmatch = [xelev[i] for i in index]

    # find dtm values of y closest to mrms coords
    yelev = codtm.y
    yelev = yelev.to_numpy()

    yprecip = ds.latitude
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

    return noband