import numpy as np
from skimage.measure import label, regionprops
import rasterio
import rasterio.features 

# This can be any array

class BoundaryBoxProcessor():
    def __init__(self):
        pass
    
    def fill_bounding_boxes(self, mask, file):
        l = label(mask)
        for s in regionprops(l):
            mask[s.slice] = 1
        src = rasterio.open(file)
        results = ({'properties': {'raster_val': v}, 'geometry': s}   for i, (s, v) in enumerate(shapes = rasterio.features.shapes(src, mask=mask, transform = src.transform)))
        print(results)
        return mask