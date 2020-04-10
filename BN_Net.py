from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Activation, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, GlobalMaxPooling2D, BatchNormalization, LeakyReLU, UpSampling2D
from keras.models import Sequential
from keras import optimizers
import keras

import numpy as np
import rasterio
import glob

input_shape = (544, 544, 3)
droprate = 0.25


def downBlock(input, filters):
    conv_1 = Conv2D(filters=filters,

                    kernel_size=(3, 3),
                    padding="same",
                    activation='relu',
                    data_format='channels_last')(input)
    bn_1 = BatchNormalization(axis=-1, momentum=0.99,
                              epsilon=0.001, center=True, scale=True,
                              beta_initializer='zeros', gamma_initializer='ones',
                              moving_mean_initializer='zeros', moving_variance_initializer='ones',
                              beta_regularizer=None, gamma_regularizer=None,
                              beta_constraint=None, gamma_constraint=None)(conv_1)
    conv_2 = Conv2D(filters=filters*2,
                    kernel_size=(3, 3),
                    padding="same",
                    activation='relu',
                    data_format='channels_last')(bn_1)
    drop = Dropout(droprate)(conv_2)  # 3
    bn_2 = BatchNormalization(axis=-1, momentum=0.99,
                              epsilon=0.001, center=True, scale=True,
                              beta_initializer='zeros', gamma_initializer='ones',
                              moving_mean_initializer='zeros', moving_variance_initializer='ones',
                              beta_regularizer=None, gamma_regularizer=None,
                              beta_constraint=None, gamma_constraint=None)(drop)
    return MaxPooling2D(pool_size=2)(bn_2), drop


def central_block(input, filters):
    # The central conv-block is a 3 × 3 convolutional layer with 384 kernels followed by a LeakyReLU activation function and BN layer.
    conv_1 = Conv2D(filters=filters,
                    kernel_size=(3, 3),
                    padding="same",
                    activation='relu',
                    data_format='channels_last')(input)
    return BatchNormalization(axis=-1, momentum=0.99,
                              epsilon=0.001, center=True, scale=True,
                              beta_initializer='zeros', gamma_initializer='ones',
                              moving_mean_initializer='zeros', moving_variance_initializer='ones',
                              beta_regularizer=None, gamma_regularizer=None,
                              beta_constraint=None, gamma_constraint=None)(conv_1)


def up_block(input, connection, filters):
    conv_1 = Conv2D(filters=filters,
                    kernel_size=(1, 1),
                    padding="same",
                    activation='relu',
                    data_format='channels_last')(input)
    up = UpSampling2D(size=2, interpolation="bilinear")(conv_1)
    con = Concatenate()([up, connection])

    conv_2 = Conv2D(filters=filters,
                    kernel_size=(3, 3),
                    activation='relu',
                    padding="same")(con)
    drop = Dropout(droprate)(conv_2)  # 3
    bn_2 = BatchNormalization(axis=-1)(drop)
    conv_2 = Conv2D(filters=filters,
                    kernel_size=(3, 3),
                    padding="same",
                    activation='relu',
                    data_format='channels_last')(bn_2)
    return BatchNormalization(axis=-1)(conv_2)


inputs = Input(shape=input_shape)

down_block_1, bn_1 = downBlock(inputs, 24)
down_block_2, bn_2 = downBlock(down_block_1, 48)
down_block_3, bn_3 = downBlock(down_block_2, 96)
down_block_4, bn_4 = downBlock(down_block_3, 192)

central_block = central_block(down_block_4, 384)


up_block_1 = up_block(central_block, bn_4, 192)
up_block_2 = up_block(up_block_1, bn_3, 96)
up_block_3 = up_block(up_block_2, bn_2, 48)
up_block_4 = up_block(up_block_3, bn_1, 24)

segm = Conv2D(filters=1,
              kernel_size=(1, 1),
              padding="same",
              activation="sigmoid",
              data_format='channels_last')(up_block_4)
model = Model(inputs=inputs, outputs=segm)
model.summary()

# 4160
train_X = np.empty((500, 544, 544, 3))
train_Y = np.empty((500, 544, 544, 1))

file_list = glob.glob("./data/train_val/train/*_mask.tif")

for idx, file in enumerate(file_list[0:500]):
    mask = rasterio.open(file)
    mask_data = mask.read()

    min_dem = np.min(mask_data)
    max_dem = np.max(mask_data)
    if(min_dem+max_dem > 0):
        processed = np.array(
            list(map(lambda x: ((x-min_dem)/(max_dem-min_dem))*255, mask_data)))
    else:
        processed = mask_data

    input_data = []

    for i in range(3):
        input_data.append(processed[0, 0:544, 0:544])
    input_data = np.array(input_data)

    input_data = np.where(input_data < 0, 0, input_data)
    input_data = np.reshape(input_data, (544, 544, 3))

    out_data = input_data.astype(np.uint8)
    out_data = np.where(out_data > 0, 1, out_data)

    train_X[idx] = input_data/255
    train_Y[idx] = np.reshape(out_data[:, :, 1], (544, 544, 1))


file_list = glob.glob("./data/train_val/validation/*_mask.tif")

val_X = np.empty((100, 544, 544, 3))
val_Y = np.empty((100, 544, 544, 1))

for idx, file in enumerate(file_list[0:100]):
    mask = rasterio.open(file)
    mask_data = mask.read()

    min_dem = np.min(mask_data)
    max_dem = np.max(mask_data)

    if(min_dem+max_dem > 0):
        processed = np.array(
            list(map(lambda x: ((x-min_dem)/(max_dem-min_dem))*255, mask_data)))
    else:
        processed = mask_data
    input_data = []

    for i in range(3):
        input_data.append(processed[0, 0:544, 0:544])
    input_data = np.array(input_data)

    input_data = np.where(input_data < 0, 0, input_data)
    input_data = np.reshape(input_data, (544, 544, 3))

    out_data = np.where(input_data > 0, 1, input_data)

    val_X[idx] = input_data/255
    val_Y[idx] = np.reshape(out_data[:, :, 1], (544, 544, 1))


optimizer = keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=optimizer,  # Optimizer
              # Loss function to minimize
              loss=keras.losses.BinaryCrossentropy(),
              # List of metrics to monitor
              metrics=['binary_crossentropy'])
model.fit(train_X, train_Y, batch_size=20,
          epochs=1, validation_data=(val_X, val_Y))


# file_list = glob.glob("./data/train_val/test/*_mask.tif")


# #1904
# val_X = np.empty((100, 544, 544, 3))
# val_Y = np.empty((100, 544, 544, 1))

# for idx, file in enumerate(file_list[0:100]):
#     mask = rasterio.open(file)
#     mask_data = mask.read()

#     min_dem = np.min(mask_data)
#     max_dem = np.max(mask_data)
#     try:
#         processed = np.array(
#             list(map(lambda x: ((x-min_dem)/(max_dem-min_dem))*255, mask_data)))
#     except Exception as e:
#         processed = mask_data
#     input_data = []

#     for i in range(3):
#         input_data.append(processed[0, 0:544, 0:544])
#     input_data = np.array(input_data)

#     input_data = np.where(input_data < 0, 0, input_data)
#     input_data = np.reshape(input_data, (544, 544, 3))

#     out_data = np.where(input_data > 0, 1, input_data)

#     val_X[idx] = input_data/255
#     val_Y[idx] = np.reshape(out_data[:, :, 1], (544, 544, 1))
