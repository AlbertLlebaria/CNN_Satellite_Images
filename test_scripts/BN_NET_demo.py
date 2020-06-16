from keras.models import Model
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Activation, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, GlobalMaxPooling2D, BatchNormalization, LeakyReLU, UpSampling2D
from keras.models import Sequential,load_model
import keras
from matplotlib import pyplot
import numpy as np
import rasterio
import glob
import os

np.seterr(divide='ignore', invalid='ignore')

MODEL_DIR = './weights'
MODEL_DIR = './model'
WEIGHTS_FILE ="weights-improvement-01-0.95.hdf5"
CHECK_POINT_PATH = "./train_ckpt"

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
    drop = Dropout(DRORATE)(conv_2)  # 3
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
    drop = Dropout(DRORATE)(conv_2)  # 3
    bn_2 = BatchNormalization(axis=-1)(drop)
    conv_2 = Conv2D(filters=filters,
                    kernel_size=(3, 3),
                    padding="same",
                    activation='relu',
                    data_format='channels_last')(bn_2)
    return BatchNormalization(axis=-1)(conv_2)


inputs = Input(shape=INPUT_SHAPE)

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

file_list = glob.glob("./data/train_val/test/*_mask.tif")
test_X = np.empty((1000, 544, 544, 1))
test_Y = np.empty((1000, 544, 544, 1))

for idx, file in enumerate(file_list[0:1000]):
    rgb = rasterio.open(file.replace("mask","elevation"))
    # rgb_data = rgb.read([1])
    input_data = rgb.read([1])

    mask = rasterio.open(file)
    mask_data = mask.read()

    out_data = mask_data.astype(np.uint8)
    out_data = np.where(out_data <= 0, 0, out_data)
    out_data = np.where(out_data > 0, 1, out_data)

    # input_data = np.reshape(rgb_data, (544, 544, 1))
    elevation_max = np.max(input_data)
    input_data = (((input_data) * (255 - 0)) / (elevation_max )) 
    input_data = np.reshape(input_data, (544, 544, 1))
    
    test_X[idx] = input_data/255
    test_Y[idx] = np.reshape(out_data,(544,544,1))


weights = glob.glob(os.path.join(CHECK_POINT_PATH,"*.hdf5"))
weights.sort()
for weightFile in weights:
    try:
        model.load_weights(os.path.join(CHECK_POINT_PATH,weightFile))
        print(f"loaded {weightFile}")
    except:
        print(f"failed loading {weightFile}")
        pass
    scores = model.evaluate(test_X, test_Y, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# out = model.predict(test_X)
# for idx, prediction in enumerate(out):
#     classes = np.where(prediction < 0.57, 0, 255)
    
#     classes =  np.reshape(classes, (544,544))
#     img =  np.reshape(test_X[idx], (544,544))
#     truth =  np.reshape(test_Y[idx], (544,544))

#     pyplot.imshow(img, cmap='pink')
#     pyplot.title('Raster elevation')
#     pyplot.show()
#     pyplot.imshow(truth, cmap='pink')
#     pyplot.title('Truth elevation')
#     pyplot.show()  
#     pyplot.imshow(classes, cmap='pink')
#     pyplot.title('Predited elevation')
#     pyplot.show() 