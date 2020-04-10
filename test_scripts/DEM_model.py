from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer,GlobalMaxPooling2D
from keras.models import Sequential
from keras import optimizers
import keras
import numpy as np
import rasterio
import glob
from modules import utils
import scipy.misc

# We use ResNet50 deep learning model as the pre-trained model for feature extraction for Transfer Learning.
# To implement Transfer learning, we will remove the last predicting layer of the pre-trained ResNet50 model
#  and replace them with our own predicting layers. FC-T1 and FC_T2 as shown below
# Weights of ResNet50 pre-trained model is used as feature extractor
# Weights of the pre-trained model are frozen and are not updated during the training


# We do not want to load the last fully connected layers which act as the classifier. We accomplish that by using “include_top = False”.
# We do this so that we can add our own fully connected layers on top of the ResNet50 model for our task-specific classification.
# We freeze the weights of the model by setting trainable as “False”. This stops any updates to the pre-trained weights during training
# We do not want to train ResNet layers as we want to leverage the knowledge learned by the deep neural network trained from the previous
# data set which in our case is “imagenet”

input_shape = (200, 200, 3)
tile_width = input_shape[0]
tile_height = input_shape[1]

restnet = ResNet50(include_top=False, weights='imagenet',
                   input_shape=input_shape)

output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)

restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False

model = Sequential()

model.add(Conv2D(filters=10, 
             kernel_size=(3, 3), 
             input_shape=(None, None, 1),
             padding="same",
             data_format='channels_last'))
model.add(GlobalMaxPooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.summary()


file_list = glob.glob("./data/train_val/*_mask.tif")

train = file_list[0:40]
val = file_list[100:10]

print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
print("Creating training set ")
train_x_data = np.empty((40,  200, 200, 3))
train_y_data = np.empty((10,  200, 200, 1))

idx = 0
for file in train:
    try:

        DEM_src = rasterio.open(file)
        for window, transform in utils.get_tiles(DEM_src, tile_width, tile_height):
            DEM_data = DEM_src.read(window=window)

            min_dem = np.min(DEM_data)
            max_dem = np.max(DEM_data)

            max_dem = 1 if max_dem == 0 else max_dem
            try:
                processed = np.array(
                    list(map(lambda x: ((x-min_dem)/(max_dem-min_dem))*255, DEM_data)))
            except Exception as e:
                processed = DEM_data
            input_data = []
        
            for i in range(3):
                input_data.append(processed[0, 0:200, 0:200])
            input_data = np.array(input_data)

            mask_data = np.where(mask_data > 0, 1, DEM_data)
            mask_data = np.where(mask_data < 0, 0, mask_data)
            

            print(mask_data.shape)
            print()
            # out_labels = np.reshape(mask_data, (200, 200, 1))
            # input_data = np.reshape(input_data, (200, 200, 3))

            # input_data = input_data / 255
            
            # train_x_data[idx] = input_data
            # train_y_data[idx] = out_labels
            # idx += 1

    except Exception as e:
        print(e)
print("Taining Set finished ")
print("Creating validation set ")

