import numpy as np
from skimage.measure import label, regionprops
import rasterio
import rasterio.features
import rasterio.mask
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, asShape
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import fiona
import os

TMP_DIR = 'tmp'


class BoundaryBoxProcessor():
    def __init__(self):
        pass

    def fill_bounding_boxes(self, mask, file):
        bool_mask = mask == 255
        bool_mask = np.expand_dims(bool_mask, axis=0)

        src = rasterio.open(file)
        data = src.read()
        meta = src.meta

        data[~bool_mask] = 0
        with rasterio.open(os.path.join(TMP_DIR, "tmp.tif"), "w", **meta) as dest:
            dest.write(data)
            return True
        return False

    def create_out_raster(self, mask, src):
        mask = mask.astype(np.float32)
        out_meta = src.meta
        with rasterio.open(os.path.join(TMP_DIR, "tmp.tif"), "w", **out_meta) as dest:
            dest.write(mask)
            return True
        return False

    def old_bounding_boxes(self, mask, file):

        bool_mask = mask == 255
        input_file = rasterio.open(file)
        is_saved_file = self.create_out_raster(
            np.expand_dims(mask, axis=0), input_file)
        src = rasterio.open('tmp.tif')

        results: {}
        if(is_saved_file):
            results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(rasterio.features.shapes(
                src.read(), mask=np.expand_dims(bool_mask, axis=0), connectivity=4, transform=src.transform)))
            with fiona.open(
                'test.shp', 'w',
                driver='Shapefile',
                crs=src.crs,
                schema={'properties': [('raster_val', 'int')],
                        'geometry': 'Polygon'}) as dst:
                dst.writerecords(results)
        return results

    def merge_adjacent_polgons(self, features, crs):
        world = gpd.read_file('test.shp')

        dissolved = world.dissolve(by='raster_val', aggfunc='first')
        dissolved.to_file("countries.shp")
        return dissolved

    def bbox(self, mask):
        l = label(mask)
        for s in regionprops(l):
            mask[s.slice] = 1
        mask = mask == 1
