import numpy as np 
import rasterio
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.util import montage as montage2d

class ElevationProcessor():

    def __init__(self):
        pass

    # ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    def rle_encode(self,img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle_decode(self,mask_rle, shape=(544, 544)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T  # Needed to align to RLE direction

    def masks_as_image(self,in_mask_list, all_masks=None):
        # Take the individual ship masks and create a single mask array for all ships
        if all_masks is None:
            all_masks = np.zeros((544, 544), dtype = np.int16)
        #if isinstance(in_mask_list, list):
        for mask in in_mask_list:
            if isinstance(mask, str):
                all_masks += self.rle_decode(mask)
        return np.expand_dims(all_masks, -1) 

    def extract_elevation_from_mask(self, mask, raster):
        print(np.max(mask))
        mask = np.where(mask  == 255  ,1, mask)
        mask_0 = self.masks_as_image([mask])
        #
        # 
        print(np.max(mask_0))
        lbl_0 = label(mask_0) 
        props = regionprops(lbl_0)
        for prop in props:
            print('Found bbox', prop.bbox)
