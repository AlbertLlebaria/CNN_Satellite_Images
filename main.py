from BN_Net import BN_NET
from BoundingBoxGenerator import BoundaryBoxProcessor
from ElevationProcessor import ElevationProcessor
from skimage.measure import label, regionprops
import numpy as np
import rasterio
from matplotlib import pyplot
import os
import glob

WEIGHT_FILE="train_ckpt/weights-improvement-02-0.93.hdf5"

def main():
    bn_net_model = BN_NET()
    bn_net_model.load_weights(WEIGHT_FILE)
    bbox_gen = BoundaryBoxProcessor()
    elevation_proc =  ElevationProcessor()

    file_list = glob.glob("./data/train_val/test/*_mask.tif")
    test_X = np.empty((3, 544, 544, 1))
    test_Y = np.empty((3, 544, 544, 1))
    rasters = []

    for idx, file in enumerate(file_list[0:3]):
                elevation = rasterio.open(file.replace("mask","elevation"))
                input_data = elevation.read([1])

                mask = rasterio.open(file)
                mask_data = mask.read()

                out_data = mask_data.astype(np.uint8)
                out_data = np.where(out_data <= 0, 0, out_data)
                out_data = np.where(out_data > 0, 1, out_data)
                input_data = np.reshape(input_data, (544, 544, 1))
         
                test_X[idx] = input_data/255
                test_Y[idx] = np.reshape(out_data,(544,544,1))

                rasters.append(elevation)

    out = bn_net_model.predict(test_X,test_Y, False)
    for idx, res in enumerate(out):
        print(bbox_gen.fill_bounding_boxes(res, rasters[idx]))


if __name__ == '__main__':
    main()