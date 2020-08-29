from BN_Net import BN_NET
from BN_Net_leaky import BN_NET as BN_Net_leaky

from boundary_box_generator import BoundaryBoxProcessor
from elevation_processor import ElevationProcessor
from skimage.measure import label, regionprops
import numpy as np
import rasterio
from matplotlib import pyplot
import os
import glob
import pandas as pd
import geopandas as gpd
from pathlib import Path


WEIGHT_FILE = "train_ckpt/leaky/weights-improvement-03-0.919.hdf5"
OUT_FILE_SHP = 'prediction_{}.shp'
OUT_DIR = './predicted'
OUT_FILE_RASTER = 'prediction_{}.tif'
    

def main():
    bn_net_model = BN_Net_leaky()
    bn_net_model.load_weights(WEIGHT_FILE)
    bbox_gen = BoundaryBoxProcessor()
    elevation_proc = ElevationProcessor()
    ranges = [0,1000,2000,3000]

    file_list = glob.glob("data/train_val/train/*_mask.tif")
    test_X = np.empty((1000, 544, 544, 1))
    test_Y = np.empty((1000, 544, 544, 1))
    for r in ranges:
        rasters = []
        for idx, file in enumerate(file_list[r:r+1000]):
            elevation = rasterio.open(file.replace("mask", "elevation"))
            input_data = elevation.read([1])

            mask = rasterio.open(file)
            mask_data = mask.read()

            out_data = mask_data.astype(np.uint8)
            out_data = np.where(out_data <= 0, 0, out_data)
            out_data = np.where(out_data > 0, 1, out_data)
            input_data = np.reshape(input_data, (544, 544, 1))

            test_X[idx] = input_data
            test_Y[idx] = np.reshape(out_data, (544, 544, 1))

            rasters.append(file)

        out = bn_net_model.predict(test_X, test_Y, False)
        predicted_shapes = []

        for idx, res in enumerate(out):
            bbox_gen.fill_bounding_boxes(res, rasters[idx])
            out_shape = elevation_proc.extract_elevation(
                rasters[idx], res, os.path.join(OUT_DIR, OUT_FILE_RASTER.format(idx)))
            predicted_shapes.append( gpd.read_file(out_shape))

        gdf = gpd.GeoDataFrame(pd.concat(predicted_shapes))
        gdf.to_file(os.path.join(OUT_DIR,f'merged_{r}.shp'))

if __name__ == '__main__':
    main()