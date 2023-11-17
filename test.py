# %%
import numpy as np
import pandas as pd
import xarray as xr

array_3d = np.random.rand(10, 10, 70000)

df = xr.DataArray(array_3d)

mean_over_dims = df.mean(dim=['dim_0', 'dim_1'])

print(mean_over_dims)
