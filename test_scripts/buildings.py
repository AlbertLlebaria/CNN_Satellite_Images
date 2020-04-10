
import rasterio
import geopandas as gpd
import numpy as np
import glob
import os
from modules import utils

from rasterio.plot import show
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats
from shapely.geometry import Point
from modules.dip import dip, plot_dip


def process_building_rasters():
    count = 0

    with rasterio.open(f'{DATADIR}geo/y.tif') as rgb:

        with rasterio.open(f'{DATADIR}geo/y.elevation.tif') as DEM:

            rgb_meta = rgb.profile.copy()
            dem_meta = DEM.profile.copy()

            train_roof = gpd.read_file(
                f'filtred_buildings/test.shp', bbox=DEM.bounds)

            for index, row in train_roof.iterrows():
                try:
                    window = rasterio.features.geometry_window(
                        rgb, [row['geometry']], pad_x=10, pad_y=10,north_up=False, pixel_precision=1)

                    rgb_meta['width'], rgb_meta['height'] = window.width, window.height
                    rgb_meta["transform"] = rasterio.windows.transform(
                        window, rgb.transform)

                    dem_meta['width'], dem_meta['height'] = window.width, window.height
                    dem_meta["transform"] = rasterio.windows.transform(
                        window, DEM.transform)

                    RGB_bands = rgb.read(window=window)
                    DEM_data = DEM.read(window=window)

                    with rasterio.open(os.path.join(output_dir, building_rgb.format(count)), 'w', **rgb_meta) as dst:
                        dst.write(RGB_bands)
                    with rasterio.open(os.path.join(output_dir, building_elevation.format(count)), 'w', **dem_meta) as dst:
                        dst.write(DEM_data)

                    count += 1
                    print("Finished building ", count)
                except Exception as e:
                    print(e)



def divide_buildings():
    train_roof = gpd.read_file(
        f'data/shapes/CPH_Buildings_Subset.shp', crs="EPSG:3044")

    areas = []
    out = []

    for idx, row in train_roof.iterrows():
        areas.append(row["geometry"].area)
    mean = np.mean(areas)

    for idx, row in train_roof.iterrows():
        if(row["geometry"].area >= mean):
            out.append(row["geometry"])

    gpd.GeoDataFrame({"geometry": out}).to_file("test.shp")
