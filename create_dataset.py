import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs
import numpy as np
import glob
from keras.layers import Input, Conv2D
from keras.models import Model
from rasterio.windows import Window
import os
import utils 


DATADIR = 'data/'

RESHAPED = 16000


TRAIN_WINDOW = Window(0, 0, 16000, 16000)
VAL_WINDOW = Window(0, 16000, 16000, 16000)



for file in glob.glob(f'{DATADIR}geo/*.tif'):
with rasterio.open(f'{DATADIR}geo/CPH_k.tif') as src:
    train_data = src.read(window=TRAIN_WINDOW)
    val_data = src.read(window=VAL_WINDOW)
    data_profile = src.profile.copy()
    print(train_data.shape)
    #write training data    
    train_profile = data_profile.copy()
    train_profile.update({'height': TRAIN_WINDOW.height,
                        'width': TRAIN_WINDOW.width,
                        'transform': rasterio.windows.transform(TRAIN_WINDOW, data_profile['transform'])
                        })

    with rasterio.open(os.path.join(DATADIR, 'train_val/cph_k_train.tif'), 'w', **train_profile) as dst:
        dst.write(train_data)

    #write test data
    val_profile = data_profile.copy()
    val_profile.update({'height': VAL_WINDOW.height,
                        'width': VAL_WINDOW.width,
                        'transform': rasterio.windows.transform(VAL_WINDOW, data_profile['transform'])
                    }
                    )
    with rasterio.open(os.path.join(DATADIR, 'train_val/cph_k_val.tif'), 'w', **val_profile) as dst:
        dst.write(val_data)

with rasterio.open(f'{DATADIR}geo/mask_k.tif') as src:
    train_mask = src.read(window=TRAIN_WINDOW)
    val_mask = src.read(window=VAL_WINDOW)
    train_mask = np.where(train_mask > 0, 255, train_mask) 
    val_mask = np.where(val_mask > 0, 255, val_mask) 

    data_profile = src.profile.copy()

    #write training data
    train_profile = data_profile.copy()
    train_profile.update({'height': TRAIN_WINDOW.height,
                        'width': TRAIN_WINDOW.width,
                        'transform': rasterio.windows.transform(TRAIN_WINDOW, data_profile['transform'])
                        }
                        )

    with rasterio.open(os.path.join(DATADIR, 'train_val/mask_k_train.tif'), 'w', **train_profile) as dst:
        dst.write(train_mask)

    #write test data
    val_profile = data_profile.copy()
    val_profile.update({'height': VAL_WINDOW.height,
                        'width': VAL_WINDOW.width,
                        'transform': rasterio.windows.transform(VAL_WINDOW, data_profile['transform'])
                    }
                    )
    with rasterio.open(os.path.join(DATADIR, 'train_val/mask_k_val.tif'), 'w', **val_profile) as dst:
        dst.write(val_mask)

print('created train val')
in_path = os.path.join(DATADIR, 'train_val')
out_path = os.path.join(DATADIR, 'train_patches')
input_filenames = [('cph_k_train.merged.tif', 'mask_k_train.tif')]

output_filename = 'tile_{}-{}.tif'
output_lab_filename = 'tile_labels_{}-{}.tif'

for filename in input_filenames:
    (input_filename, label_filename) = filename
    with rasterio.open(os.path.join(in_path, input_filename)) as inds:
        tile_width, tile_height = 224, 224
        meta = inds.meta.copy()
        nodata = meta['nodata']
    
        with rasterio.open(os.path.join(in_path, label_filename)) as labs:
            meta_labels = labs.meta.copy()
            labs_data = labs.read()
            labs_num = len(np.unique(labs_data))
            name_counter = 0
            for window, transform in utils.get_tiles(inds, tile_width, tile_height):
                
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                meta_labels['transform'] = transform
                meta_labels['width'], meta_labels['height'] = window.width, window.height
                data = inds.read(window=window)
                labels = labs.read(window=window)
                #if labs_num == 5:
                #    print("found 5 labs in :  " + label_filename)
                #    labels[labels == 5] = 6
                if window.width ==  tile_width and window.height == tile_height: #and not np.all(data[5, :, :] == nodata):
                    outpath = os.path.join(out_path, output_filename.format(input_filename[:2], name_counter))
                    outpath_lab = os.path.join(out_path, output_lab_filename.format(input_filename[:2], name_counter))
                    with rasterio.open(outpath, 'w', **meta) as outds:
                        outds.write(data)
                    with rasterio.open(outpath_lab, 'w', **meta_labels) as outls:
                        outls.write(labels)
                    name_counter += 1

### Follow the same procedure to create patches for validation ###

in_path = os.path.join(DATADIR, 'train_val')
out_path = os.path.join(DATADIR, 'val_patches')
input_filenames = [('cph_k_val.merged.tif', 'mask_k_val.tif')]

output_filename = 'tile_{}-{}.tif'
output_lab_filename = 'tile_labels_{}-{}.tif'

for filename in input_filenames:
    (input_filename, label_filename) = filename
    with rasterio.open(os.path.join(in_path, input_filename)) as inds:
        tile_width, tile_height = 224, 224
        meta = inds.meta.copy()
        nodata = meta['nodata']
    
        with rasterio.open(os.path.join(in_path, label_filename)) as labs:
            meta_labels = labs.meta.copy()
            labs_data = labs.read()
            labs_num = len(np.unique(labs_data))
            print(labs_num)
            name_counter = 0
            for window, transform in utils.get_tiles(inds, tile_width, tile_height):
                
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                meta_labels['transform'] = transform
                meta_labels['width'], meta_labels['height'] = window.width, window.height
                data = inds.read(window=window)
                labels = labs.read(window=window)
                #if labs_num == 5:
                #    print("found 5 labs in :  " + label_filename)
                #    labels[labels == 5] = 6
                if window.width ==  tile_width and window.height == tile_height: #and not np.all(data[5, :, :] == nodata):
                    outpath = os.path.join(out_path, output_filename.format(input_filename[:2], name_counter))
                    outpath_lab = os.path.join(out_path, output_lab_filename.format(input_filename[:2], name_counter))
                    with rasterio.open(outpath, 'w', **meta) as outds:
                        outds.write(data)
                    with rasterio.open(outpath_lab, 'w', **meta_labels) as outls:
                        outls.write(labels)
                    name_counter += 1