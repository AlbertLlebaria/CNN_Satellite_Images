import rasterio
import numpy as np
import os

with rasterio.open('data/train_val/cph_k_val.tif') as src:
    rgb_bands = src.read()
    meta = src.meta.copy()
    print(src.bounds)

with rasterio.open('data/train_val/cph_k_val.elevation.tif') as src:
    elevation_band = src.read()
    print(src.meta, meta)
    profile = src.profile.copy()
    profile.update({'count':6})

con = np.concatenate((rgb_bands, elevation_band), axis=0)

with rasterio.open(os.path.join('data', 'train_val/cph_k_val.merged.tif'), 'w', **profile) as dst:
    dst.write(con)