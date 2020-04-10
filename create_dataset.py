import rasterio
import geopandas as gpd
import numpy as np
import glob
import os
from modules import utils
import shutil

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


DATADIR = 'data/'


tile_width = 544
tile_height = 544

output_file_elevation = "tile_{}_elevation.tif"
output_file_rgb = "tile_{}_rgb.tif"
output_file_mask = "tile_{}_mask.tif"


building_elevation = 'building_{}_elevation.tif'
building_rgb = 'building_{}_rgb.tif'
building_mask = 'building_{}_mask.tif'


output_dir = os.path.join(DATADIR, 'train_val')


def procces__rgb_and_dem():
    count = 0

    with rasterio.open(f'{DATADIR}geo/y.tif') as rgb:

        with rasterio.open(f'{DATADIR}geo/y.elevation.tif') as DEM:

            rgb_meta = rgb.profile.copy()
            dem_meta = DEM.profile.copy()

            for window, transform in utils.get_tiles(rgb, tile_width, tile_height):

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

                print(f'Finished {output_file_rgb.format(count)}')


def create_mask():
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
                f'filtred_buildings/test.shp', bbox=src.bounds)

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
                out_image = np.full((1, 544, 544), 0.0)
                out_meta = src.meta
                out_image = out_image.astype(np.float32)
            with rasterio.open(os.path.join(output_dir, output_file_mask.format(count)), "w", **out_meta) as dest:
                dest.write(out_image)

            print(f'Finished {building_mask.format(count)}')


def normalize_roofs():
    file_list = glob.glob("./data/train_val/*_mask.tif")
    for file in file_list:
        file_name = file[file.rfind("/")+1::]
        count = int(file_name[file_name.find("_")+1:file_name.rfind("_")])

        src = rasterio.open(file)
        elevation = src.read()
        out_meta = src.meta.copy()

        train_roof = gpd.read_file(
            f'filtred_buildings/test.shp', bbox=src.bounds)

        for index, row in train_roof.iterrows():
            try:
                window = rasterio.features.geometry_window(
                    src, [row['geometry']], pad_x=30, pad_y=30, pixel_precision=1)
                building = []
                mask_b = []
                ground = min(src.read([1], window=window).flatten())

                # PLOT STUFF
                xs = []
                ys = []
                zs = []

                # TESTING STUFF

                # rgb_file = file.replace("elevation", "rgb")
                # mask_file = file.replace("elevation", "mask")

                # window2 = rasterio.features.geometry_window(
                #     src, [row['geometry']], pad_x=25, pad_y=25,pixel_precision=1)
                # show(rasterio.open(rgb_file).read([1, 2, 3], window=window2))
                # show(rasterio.open(mask_file).read([1], window=window2))

                building = []
                coords = []

                for x in range(window.row_off, window.row_off+window.height):
                    for y in range(window.col_off, window.col_off+window.width):
                        x_coord, y_coord = rasterio.transform.xy(
                            src.transform, x, y)
                        p = Point(x_coord, y_coord)
                        # IF ITS INSIDE GEOMETRY
                        if(p.within(row["geometry"]), p, row["geometry"]):
                            # IF ITS HIGHER THAN THE GROUND
                            if(elevation[0][x][y] > ground+2):
                                # zs.append(elevation[0][x][y])
                                # xs.append(x)
                                # ys.append(y)

                                building.append(elevation[0][x][y])
                                coords.append([x, y])

                if(len(building) > 0):
                    try:
                        p_value = dip(np.msort(building).astype(np.float64))
                        print(p_value[0])

                        if(p_value[0] >= 0.005):
                            print("UNIMODAL")

                            # fig = plt.figure()
                            # ax = fig.add_subplot(111, projection='3d')
                            # ax.scatter(xs, ys, building, marker='o')
                            # plt.show()

                            mda = stats.median_absolute_deviation(building)
                            diffs = list(map(lambda x: abs(x-mda), building))
                            z_scores = list(
                                map(lambda x: (0.6745*x)/mda, diffs))

                            for idx, z in enumerate(z_scores):
                                if z <= 3.0:
                                    building[idx] = building[idx]+z

                            for idx, b in enumerate(building):
                                xc = coords[idx][0]
                                yc = coords[idx][1]

                                elevation[0][xc][yc] = b

                        elif(p_value[0] < 0):
                            print("NIFLI")
                        else:
                            print("BIMODAL")

                    except Exception as e:
                        print("BUILDING HAS NO DATA", e, len(building))
                    else:
                        d = src.read(window=window)

            except rasterio.errors.WindowError as err:
                print(err, "window")

        with rasterio.open(os.path.join(output_dir, output_file_elevation.format(count)), "w", **out_meta) as dest:
            dest.write(elevation)
        print("finished ", file)


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
       
        shutil.move(file, os.path.join(train_dir, file_name))
        shutil.move(rgb_file, os.path.join(train_dir, rgb_file_aux))

        print("Moved train", file_name)

    pointer += train

    for file in file_list[pointer:pointer+validation]:
        file_name = file[file.rfind("/")+1::]
        rgb_file_aux = file_name.replace("mask", "rgb")
        rgb_file = file.replace("mask", "rgb")
        
        shutil.move(file, os.path.join(val_dir, file_name))
        shutil.move(rgb_file, os.path.join(val_dir, rgb_file_aux))

        print("Moved validation", file_name)

    pointer += validation

    for file in file_list[pointer:pointer+test]:
        file_name = file[file.rfind("/")+1::]
        rgb_file_aux = file_name.replace("mask", "rgb")
        rgb_file = file.replace("mask", "rgb")
        
        shutil.move(file, os.path.join(test_dir, file_name))
        shutil.move(rgb_file, os.path.join(test_dir, rgb_file_aux))

        print("Moved test", file_name)

procces__rgb_and_dem()
# # print("normalizing roofs")
# # normalize_roofs()
# print("Creating mask")
create_mask()
split_data_set()
