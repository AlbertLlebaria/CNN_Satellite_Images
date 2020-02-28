import tensorflow as tf
from rasterio import windows
import rasterio
from itertools import product
import numpy as np
import geopandas as gpd
import rasterio.mask

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def get_tiles(ds, width=224, height=224):
    nols, nrows = ds.meta['width'], ds.meta['height']
    # Control the level of overlap
    offsets = product(range(0, nols, width//2), range(0, nrows, height//2))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off,
                                width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def iou_metric(y_true_in, y_pred_in):
    labels = y_true_in
    y_pred = y_pred_in

    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(),
                           bins=([0, 0.5, 1], [0, 0.5, 1]))

    intersection = temp1[0]

    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    iou = intersection / union
    return iou

# create a mak


# with rasterio.open('data/geo/cph_ama.tif') as src:

#     # Read the dataset's valid data mask as a ndarray.
#     raster_data = src.read()
#     raster_metadata = src.meta.copy()
#     print(src.bounds, raster_data.shape, raster_metadata)

#     train_roof = gpd.read_file(
#         f'data/shapes/CPH_Buildings_Subset.shp', bbox=src.bounds)
#     print(train_roof)

#     out_image, out_transform = rasterio.mask.mask(
#         src, train_roof.geometry, crop=False)
#     out_meta = src.meta

#     out_meta.update({"driver": "GTiff",
#                   "height": out_image.shape[1],
#                   "width": out_image.shape[2],
#                   "transform": out_transform})

#     with rasterio.open("mask_ama.tif", "w", **out_meta) as dest:
#       dest.write(out_image)


