from modules.geofolki.algorithm import GEFolki
from modules.geofolki.tools import wrapData
from itertools import product
from modules import utils
from fiona.crs import from_epsg

import rasterio
import geopandas as gpd
import numpy as np
import glob
import os
import geopandas as gpd


DATADIR = 'data/'


tile_width = 2000
tile_height = 2000
output_file_elevation = "tile_{}_elevation.tif"
output_file_rgb = "tile_{}_rgb.tif"

output_file_mask = "tile_{}_mask.tif"

output_dir = os.path.join(DATADIR, 'train_val')


def procces__rgb_and_dem():
    with rasterio.open(f'{DATADIR}geo/y.tif') as rgb:

        with rasterio.open(f'{DATADIR}geo/y.elevation.tif') as DEM:

            rgb_meta = rgb.profile.copy()
            dem_meta = DEM.profile.copy()

            for window, transform in utils.get_tiles(rgb, tile_width, tile_height):
                print(window)
                mask_roofs = gpd.read_file(
                    f'data/shapes/CPH_Buildings_Subset.shp', bbox=rasterio.windows.bounds(window, transform))

                rgb_meta['transform'] = transform
                rgb_meta['width'], rgb_meta['height'] = window.width, window.height

                dem_meta['transform'] = transform
                dem_meta['width'], dem_meta['height'] = window.width, window.height

                RGB_bands = rgb.read(window=window)
                DEM_data = DEM.read(window=window)

                with rasterio.open(os.path.join(output_dir, output_file_rgb.format(count)), 'w', **rgb_meta) as dst:
                    dst.write(RGB_bands)
                with rasterio.open(os.path.join(output_dir, output_file_elevation.format(count)), 'w', **dem_meta) as dst:
                    dst.write(DEM_data)

                count += 1


def procces__mask():
    file_list = glob.glob("./data/train_val/*_elevation.tif")
    file_list.sort()
    for file in file_list:
        file_name = file[file.rfind("/")+1::]
        count = int(file_name[file_name.find("_")+1:file_name.rfind("_")])

        with rasterio.open(file) as src:
            # Read the dataset's valid data mask as a ndarray.
            raster_data = src.read()
            raster_metadata = src.meta.copy()

            train_roof = gpd.read_file(
                f'data/shapes/CPH_Buildings_Subset.shp', bbox=src.bounds)

            if(len(train_roof) > 0):
                out_image, out_transform = rasterio.mask.mask(
                    src, train_roof.geometry, crop=False)
                out_meta = src.meta

                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
                out_image = np.where(out_image < 0.0, 0.0, out_image)

            else:
                out_image = np.full((1, 2000, 2000), 0.0)
                out_meta = src.meta
            with rasterio.open(os.path.join(output_dir, output_file_mask.format(count)), "w", **out_meta) as dest:
                dest.write(out_image)

            print(f'Finished {output_file_mask.format(count)}')


def corregister_rasters():
    count = 0
    rgb_file = output_file_rgb.format(count)
    mask_file = output_file_mask.format(count)
    with rasterio.open(rgb_file) as rgb:
        with rasterio.open(mask_file) as mask:
