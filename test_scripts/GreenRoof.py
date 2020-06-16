import rasterio
import geopandas as gpd
import numpy as np
import glob
import os
from modules import utils
import shutil
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats
from shapely.geometry import Point
from modules.dip import dip, plot_dip
import random


# from modules.fitlers import rank
# from modules.geofolki.algorithm import GEFolki
# from modules.geofolki.algorithm import EFolki


DATADIR = './data/NY/'
DST_CRS = 'EPSG:4326'

tile_width = 500
tile_height = 500

output_file_elevation = "tile_{}_elevation.tif"
output_file_rgb = "tile_{}_rgb.tif"
output_file_mask = "tile_{}_mask.tif"


building_elevation = 'building_{}_elevation.tif'
building_rgb = 'building_{}_rgb.tif'
building_mask = 'building_{}_mask.tif'
building_mask_rgb = 'building_{}_rgbmask.tif'


output_dir = os.path.join(DATADIR, 'processed')


def transform_tiles():
    file_list = glob.glob(os.path.join(DATADIR,'Manhatan',"*.jp2"))
    file_list.sort()
    count = 0
    for file in file_list:
        # try:
            with rasterio.open(file) as src:
                print(src.meta)
        #         transform, width, height = calculate_default_transform(
        #             src.crs, DST_CRS, src.width, src.height, *src.bounds)
        #         kwargs = src.meta.copy()
        #         kwargs.update({
        #             'crs': DST_CRS,
        #             'transform': transform,
        #             'width': width,
        #             'height': height
        #         })
        #         projected_file ="projected_{}.tif"
        #         with rasterio.open(os.path.join(DATADIR,'Manhatan',projected_file.format(count)), 'w', **kwargs) as dst:
        #             for i in range(1, src.count + 1):
        #                 reproject(
        #                     source=rasterio.band(src, i),
        #                     destination=rasterio.band(dst, i),
        #                     src_transform=src.transform,
        #                     src_crs=src.crs,
        #                     dst_transform=transform,
        #                     dst_crs=DST_CRS,
        #                     resampling=Resampling.nearest)
        #             print(f'File {projected_file.format(count)} projected')
        #             count +=1
        # except Exception as e: 
        #     print(f'File {file} could not be reprojected, {e}' )
        #     print(rasterio.open(file).meta['crs'])

def create_tiles():
            count = 0
            file_list = glob.glob(os.path.join(DATADIR,'Manhatan',"*.jp2"))
            file_list.sort()
            for rgb_file in file_list:
                rgb = rasterio.open(rgb_file)
                rgb_meta  = rgb.meta.copy()
                for window, transform in utils.get_tiles(rgb, tile_width, tile_height):

                    rgb_meta['transform'] = transform
                    rgb_meta['width'], rgb_meta['height'] = window.width, window.height

                    RGB_bands = rgb.read(window=window)

                    with rasterio.open(os.path.join(output_dir, output_file_rgb.format(count)), 'w', **rgb_meta) as dst:
                        dst.write(RGB_bands)
                    count += 1
                    print(f'Finished {output_file_rgb.format(count)}')


def create_mask():
    count = 0
    file_list = glob.glob(os.path.join(output_dir,"*.tif"))
    file_list.sort()
    for file in file_list:
        file_name = file[file.rfind("/")+1::]
        count = int(file_name[file_name.find("_")+1:file_name.rfind("_")])

        with rasterio.open(file) as src:
            # Read the dataset's valid data mask as a ndarray.
            raster_data = src.read()
            raster_metadata = src.meta.copy()
            train_roof = gpd.read_file(os.path.join(DATADIR,'shapes','Roofs','buildings.shp'), bbox=src.bounds)
            print(raster_metadata,  gpd.read_file(os.path.join(DATADIR,'shapes','Roofs','buildings.shp')).crs)
            if(len(train_roof) > 0):
                out_image, out_transform = rasterio.mask.mask(
                    src, train_roof.geometry, crop=False)
                out_meta = src.meta
                print(f'{file} has green building')

                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
                out_image = np.where(out_image < 0.0, 0.0, out_image)
                out_image = out_image.astype(np.uint8)

            else:
                out_image = np.full((4, tile_width, tile_width), 0.0)
                out_meta = src.meta
                out_image = out_image.astype(np.uint8)
            with rasterio.open(os.path.join(output_dir, output_file_mask.format(count)), "w", **out_meta) as dest:
                dest.write(out_image)

            print(f'Finished {building_mask_rgb.format(count)}')


train_dir = os.path.join(DATADIR, 'train_val/train')
test_dir = os.path.join(DATADIR, 'train_val/test')
val_dir = os.path.join(DATADIR, 'train_val/validation')


def split_data_set():
    file_list = glob.glob("./data/train_val/*_mask.tif")
    train = int(len(file_list)*0.65)
    validation = int((len(file_list)-train)*0.15)
    test = len(file_list) - train-validation
    pointer = 0

    random.shuffle(file_list)

    for file in file_list[pointer:train]:
        file_name = file[file.rfind("/")+1::]
        rgb_file_aux = file_name.replace("mask", "rgb")
        rgb_file = file.replace("mask", "rgb")
        
        elevation_file_aux = file_name.replace("mask", "elevation")
        elevation_file = file.replace("mask", "elevation")

        shutil.move(file, os.path.join(train_dir, file_name))
        shutil.move(rgb_file, os.path.join(train_dir, rgb_file_aux))
        shutil.move(elevation_file, os.path.join(train_dir, elevation_file_aux))

        print("Moved train", file_name)

    pointer += train

    for file in file_list[pointer:pointer+validation]:
        file_name = file[file.rfind("/")+1::]
        rgb_file_aux = file_name.replace("mask", "rgb")
        rgb_file = file.replace("mask", "rgb")
        
        elevation_file_aux = file_name.replace("mask", "elevation")
        elevation_file = file.replace("mask", "elevation")

        shutil.move(file, os.path.join(val_dir, file_name))
        shutil.move(rgb_file, os.path.join(val_dir, rgb_file_aux))
        shutil.move(elevation_file, os.path.join(val_dir, elevation_file_aux))

        print("Moved validation", file_name)

    pointer += validation

    for file in file_list[pointer:pointer+test]:
        file_name = file[file.rfind("/")+1::]
        rgb_file_aux = file_name.replace("mask", "rgb")
        rgb_file = file.replace("mask", "rgb")


        elevation_file_aux = file_name.replace("mask", "elevation")
        elevation_file = file.replace("mask", "elevation")

        shutil.move(file, os.path.join(test_dir, file_name))
        shutil.move(rgb_file, os.path.join(test_dir, rgb_file_aux))
        shutil.move(elevation_file, os.path.join(test_dir, elevation_file_aux))

        print("Moved test", file_name)


transform_tiles()
# create_tiles()
# print("normalizing roofs")
# normalize_roofs()
# print("Creating mask")
# create_mask()
# split_data_set()
