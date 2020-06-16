import numpy as np
from skimage.measure.label import label
from skimage.measure.regionprops import regionprops

# This can be any array

class BoundaryBoxProcessor():
    def __init__(self):
        pass
    
    def fill_bounding_boxes(self, mask):
        l = label(mask)
        for s in regionprops(l):
            x[s.slice] = 1
        return x