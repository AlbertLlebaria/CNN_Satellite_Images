import numpy as np
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.util import montage as montage2d
from osgeo import gdal
import fiona
import os

TMP_DIR = 'tmp'


class ElevationProcessor():

    def __init__(self):
        pass

    def extract_elevation(self, file, mask, out_file):
        bool_mask = mask == 255
        gdal.DEMProcessing(os.path.join(TMP_DIR, 'slope.tif'),
                           os.path.join('tmp.tif'), 'slope')

        src = rasterio.open(os.path.join(TMP_DIR, 'slope.tif'))
        data = src.read()

        classified_raster = np.copy(data)
        ground = np.where((data < 0))
        correct_slope = np.where((data > 0) & (data <= 30))
        wrong_slope = np.where(data > 30)

        classified_raster[ground] = 0
        classified_raster[correct_slope] = 1
        classified_raster[wrong_slope] = 0

        classified_raster.astype(np.int8)
        meta = src.meta

        with rasterio.open(out_file, "w", **meta) as dest:
            dest.write(classified_raster)

        src = rasterio.open(out_file)
        results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(rasterio.features.shapes(
            src.read(), mask=np.expand_dims(bool_mask, axis=0), connectivity=4, transform=src.transform)))
        with fiona.open(
            out_file.replace(".tif", ".shp"), 'w',
            driver='Shapefile',
            crs=src.crs,
            schema={'properties': [('raster_val', 'int')],
                    'geometry': 'Polygon'}) as dst:
            dst.writerecords(results)
