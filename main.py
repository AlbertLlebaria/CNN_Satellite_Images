# reading training data and labels
from segmentation_models.utils import set_trainable
from segmentation_models.metrics import iou_score
from segmentation_models.losses import jaccard_loss
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import glob
import numpy as np
import rasterio
import os
import segmentation_models as sm
from segmentation_models import Unet
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
import tensorflow as tf
from matplotlib.pyplot import imshow
from keras.utils import Sequence
from albumentations import (Compose, RandomCrop, HorizontalFlip, VerticalFlip, RandomRotate90, Transpose,
                            ShiftScaleRotate, RandomGamma, IAAEmboss, Blur, OneOf,
                            ElasticTransform, GridDistortion, OpticalDistortion, ToFloat)

from keras.layers import Input, Conv2D
from keras.models import Model
import utils
from matplotlib import pyplot as plt
import time


DATADIR = 'data/'


patch_path = os.path.join(DATADIR, 'train_patches/')

X_train = np.empty((400, 224, 224, 5), dtype=np.float32)
y_train = np.empty((400, 224, 224), dtype=np.int8)

for index, filename in enumerate(sorted(glob.glob(f'{patch_path}tile_cp*.tif'))):
    if(index >= 400):
        break
    print(filename)
    print(index)
    with rasterio.open(filename, 'r') as src:
        data = src.read([1, 2, 3, 4, 6])
        print(src.nodatavals)
        profile = src.profile
    nodata = profile['nodata']
    data = np.swapaxes(data, 0, 1)
    data = np.swapaxes(data, 1, 2)
    X_train[index, :, :, :] = data

for index, filename in enumerate(sorted(glob.glob(f'{patch_path}tile_labels_cp*.tif'))):
    if(index >= 400):
        break
    print(filename)
    print(index)
    with rasterio.open(filename, 'r') as src:
        labels = src.read()
        labels = labels.astype(np.int8)
    np.where(labels == 255, 1, labels)
    y_train[index, :, :] = labels[0, :, :]

# reading validation data and labels
patch_path = os.path.join(DATADIR, 'val_patches/')

X_val = np.empty((200, 224, 224, 5), dtype=np.float32)
y_val = np.empty((200, 224, 224), dtype=np.int8)

for index, filename in enumerate(sorted(glob.glob(f'{patch_path}tile_cp*.tif'))):
    if(index >= 200):
        break
    with rasterio.open(filename, 'r') as src:
        data = src.read([1, 2, 3, 4, 6])
        profile = src.profile
    nodata = profile['nodata']
    data = np.swapaxes(data, 0, 1)
    data = np.swapaxes(data, 1, 2)
    X_val[index, :, :, :] = data

for index, filename in enumerate(sorted(glob.glob(f'{patch_path}tile_labels_cp*.tif'))):
    if(index >= 200):
        break
    print(filename)
    print(index)
    with rasterio.open(filename, 'r') as src:
        labels = src.read()
        labels = labels.astype(np.int8)
    np.where(labels == 255, 1, labels)
    y_val[index, :, :] = labels[0, :, :]


for band in range(X_train.shape[3]):
    minX = np.min(X_train[:, :, :, band])
    maxX = np.max(X_train[:, :, :, band])
    X_train[:, :, :, band] = (X_train[:, :, :, band] - minX) / (maxX - minX)
    X_val[:, :, :, band] = (X_val[:, :, :, band] - minX) / (maxX - minX)


# sequence to load data and apply augmentation
y_val = y_val.reshape((200, 224, 224, 1))
y_train = y_train.reshape((400, 224, 224, 1))


class UnetSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, augmentations):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        data_index_min = int(idx*self.batch_size)
        data_index_max = int(min((idx+1)*self.batch_size, len(self.x)))

        indexes = self.x[data_index_min:data_index_max]
        this_batch_size = len(indexes)

        X = np.empty((this_batch_size, 224, 224, 5), dtype=np.float32)
        y = np.empty((this_batch_size, 224, 224, 1), dtype=np.uint8)

        for i, sample_index in enumerate(indexes):
            x_sample = self.x[idx, :, :, :]
            y_sample = self.y[idx, :, :]
            if self.augment is not None:
                augmented = self.augment(image=x_sample, mask=y_sample)
                image_augm = augmented['image']
                mask_augm = augmented['mask']
                X[i, :, :, :] = image_augm
                y[i, :, :, :] = mask_augm
            else:
                X[i, :, :, :] = x_sample
                y[i, :, :, :] = y_sample
        return X, y

train_augm = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    Transpose(p=0.5),
], p=1)

base_model = Unet(backbone_name='efficientnetb0',
                  encoder_weights='imagenet', encoder_freeze=True)
inp = Input(shape=(None, None, 5))
l1 = Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
out = base_model(l1)
model = Model(inp, out, name=base_model.name)


# reduces learning rate on plateau
lr_reducer = ReduceLROnPlateau(factor=0.1,
                               cooldown=10,
                               patience=10, verbose=1,
                               min_lr=0.1e-5)
# model autosave callbacks
mode_autosave = ModelCheckpoint("./weights/unet.h5",
                                monitor='val_iou_score',
                                mode='max', save_best_only=True, verbose=1, period=1)


callbacks = [lr_reducer, mode_autosave]


lre = 0.001
adam = Adam(lr=lre)
model.compile(adam, jaccard_loss, ['binary_accuracy', iou_score])

train_gen = UnetSequence(X_train, y_train, 8, augmentations=None)
test_gen = UnetSequence(X_val, y_val, 8, augmentations=None)


print(model.summary())
model.fit(X_train, y_train, epochs=2, batch_size=8,
          validation_data=(X_val, y_val))

y_pred = model.predict(X_val)
model.save_weights("./weights/unet.h5")
