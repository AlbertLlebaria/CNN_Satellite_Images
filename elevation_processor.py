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
from shapely.geometry import shape, mapping
from matplotlib import pyplot

TMP_DIR = 'tmp'
import pdb

class ElevationProcessor():

    def __init__(self):
        pass

    def extract_elevation(self, file, mask, out_file):
        prediction_mask = mask == 255

        gdal.DEMProcessing(os.path.join(TMP_DIR, 'slope.tif'),
                           os.path.join(TMP_DIR, 'tmp.tif'), 'slope')

        src = rasterio.open(os.path.join(TMP_DIR, 'slope.tif'))
        meta = src.meta        
        data = src.read()[0]

        classified_raster = np.copy(data)
        classified_raster.astype(np.int8)

        for x in range(544):
            for j in range(544):
                if( classified_raster[x][j] >=0  and classified_raster[x][j] <=30  and prediction_mask[x][j] ):
                    classified_raster[x][j] = 255
                else:
                    classified_raster[x][j] = -1

        bool_mask = classified_raster == 255

        
        classified_raster =  np.array([classified_raster])
        with rasterio.open(out_file, "w", **meta) as dest:
            dest.write(np.array([classified_raster]))

        src = rasterio.open(out_file)
        results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(rasterio.features.shapes(
            classified_raster, mask=np.expand_dims(bool_mask, axis=0), connectivity=4, transform=src.transform)))

        out_shapefile = out_file.replace(".tif", ".shp")
        tmp_out_shapefile = out_file.replace(".tif", ".tmp.shp")
        with fiona.open(
            tmp_out_shapefile, 'w',
            driver='Shapefile',
            crs=src.crs,
            schema={'properties': [('raster_val', 'int')],
                    'geometry': 'Polygon'}) as dst:
            dst.writerecords(results)
                
        return self.clean_by_area(tmp_out_shapefile,out_shapefile,src)

    def clean_by_area(self, tmp_file, out_file, src):
        with fiona.open(tmp_file, "r") as source:
            with fiona.open(
                out_file, 'w',
                driver='Shapefile',
                crs=src.crs,
                schema={'properties': [('raster_val', 'int')],
                        'geometry': 'Polygon'}) as dst:
                                    for feature in source:
                                        area = shape(feature["geometry"]).area
                                        if(area > 10.0):
                                            dst.write(feature)
        return out_file
