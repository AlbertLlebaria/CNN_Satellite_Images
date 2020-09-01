from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Activation, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, GlobalMaxPooling2D, BatchNormalization, LeakyReLU, UpSampling2D
from keras.models import Sequential
from keras import optimizers
import keras
from matplotlib import pyplot
import numpy as np
import rasterio
import glob
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import math
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from datetime import datetime
from matplotlib.colors import ListedColormap
import sys

np.seterr(divide='ignore', invalid='ignore')

MODEL_DIR = './model'
DATA_DIR = './data'
TRAIN_DIR = 'train_val/train'
VAL_DIR = 'train_val/validation'
CHECK_POINT_PATH = "train_ckpt/leaky"
WEIGHT_FILE = "train_ckpt/leaky/weights-improvement-03-0.919.hdf5"
DRORATE = 0.25
LEARNING_RATE = 2*math.pow(10, -4)
colors = ["white", "black", "red", "blue"]  # use hex colors here, if desired.
cmap = ListedColormap(colors)

class BN_NET:
    def __init__(self, drop_rate=DRORATE, learning_rate=LEARNING_RATE, model_dir=MODEL_DIR):
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.input_shape = (544, 544, 1)
        self.model_dir = model_dir
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model = self.create_model()
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        self.filepath = "./train_ckpt/leaky/v2/weights-improvement-{epoch:02d}-{val_accuracy:.3f}.hdf5"
        self.filepath_loss = "./train_ckpt/leaky/v2/weights-improvement-lss-{epoch:02d}-{val_loss:.3f}.hdf5"

    def create_model(self):
        inputs = Input(shape=self.input_shape)

        down_block_1, bn_1 = self.downBlock(inputs, 24)
        down_block_2, bn_2 = self.downBlock(down_block_1, 48)
        down_block_3, bn_3 = self.downBlock(down_block_2, 96)
        down_block_4, bn_4 = self.downBlock(down_block_3, 192)

        central_block = self.central_block(down_block_4, 384)

        up_block_1 = self.up_block(central_block, bn_4, 192)
        up_block_2 = self.up_block(up_block_1, bn_3, 96)
        up_block_3 = self.up_block(up_block_2, bn_2, 48)
        up_block_4 = self.up_block(up_block_3, bn_1, 24)

        segm = Conv2D(filters=1,
                      kernel_size=(1, 1),
                      padding="same",
                      activation="sigmoid",
                      data_format='channels_last')(up_block_4)
        model = Model(inputs=inputs, outputs=segm)
        model.summary()
        model.compile(optimizer=self.optimizer,  # Optimizer
                      # Loss function to minimize
                      loss=keras.losses.BinaryCrossentropy(),
                      # List of metrics to monitor
                      metrics=[keras.metrics.Precision(), keras.metrics.Recall(), 'accuracy'])

        return model

    def downBlock(self, input, filters):
        conv_1 = Conv2D(filters=filters,
                        kernel_size=(3, 3),
                        padding="same",
                        data_format='channels_last')(input)
        leaky_1 = LeakyReLU(alpha=0.1)(conv_1)
        bn_1 = BatchNormalization(axis=-1, momentum=0.99,
                                  epsilon=0.001, center=True, scale=True,
                                  beta_initializer='zeros', gamma_initializer='ones',
                                  moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                  beta_regularizer=None, gamma_regularizer=None,
                                  beta_constraint=None, gamma_constraint=None)(leaky_1)
        conv_2 = Conv2D(filters=filters*2,
                        kernel_size=(3, 3),
                        padding="same",
                        data_format='channels_last')(bn_1)
        leaky_2 = LeakyReLU(alpha=0.1)(conv_2)
        drop = Dropout(DRORATE)(leaky_2)  # 3
        bn_2 = BatchNormalization(axis=-1, momentum=0.99,
                                  epsilon=0.001, center=True, scale=True,
                                  beta_initializer='zeros', gamma_initializer='ones',
                                  moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                  beta_regularizer=None, gamma_regularizer=None,
                                  beta_constraint=None, gamma_constraint=None)(drop)
        return MaxPooling2D(pool_size=2)(bn_2), bn_2

    def central_block(self, input, filters):
        # The central conv-block is a 3 × 3 convolutional layer with 384 kernels followed by a LeakyReLU activation function and BN layer.
        conv_1 = Conv2D(filters=filters,
                        kernel_size=(3, 3),
                        padding="same",
                        data_format='channels_last')(input)
        leaky_1 = LeakyReLU(alpha=0.1)(conv_1)
        return BatchNormalization(axis=-1, momentum=0.99,
                                  epsilon=0.001, center=True, scale=True,
                                  beta_initializer='zeros', gamma_initializer='ones',
                                  moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                  beta_regularizer=None, gamma_regularizer=None,
                                  beta_constraint=None, gamma_constraint=None)(leaky_1)

    def up_block(self, input, connection, filters):
        conv_1 = Conv2D(filters=filters,
                        kernel_size=(1, 1),
                        padding="same",
                        data_format='channels_last')(input)
        leaky_1 = LeakyReLU(alpha=0.1)(conv_1)
        up = UpSampling2D(size=2, interpolation="bilinear")(leaky_1)
        con = Concatenate()([up, connection])

        conv_2 = Conv2D(filters=filters,
                        kernel_size=(3, 3),
                        padding="same")(con)
        leaky_2 = LeakyReLU(alpha=0.1)(conv_2)
        drop = Dropout(DRORATE)(leaky_2)  # 3
        bn_2 = BatchNormalization(axis=-1)(drop)
        conv_2 = Conv2D(filters=filters,
                        kernel_size=(3, 3),
                        padding="same",
                        data_format='channels_last')(bn_2)
        leaky_3 = LeakyReLU(alpha=0.1)(conv_2)
        return BatchNormalization(axis=-1)(leaky_3)

    def train(self):

        file_list = glob.glob(os.path.join(DATA_DIR, VAL_DIR, "*_mask.tif"))
        file_list.sort()
        # 336 files in validation directory
        val_X = np.empty((336, 544, 544, 1))
        val_Y = np.empty((336, 544, 544, 1))
        for idx, file in enumerate(file_list[0:336]):
            rgb = rasterio.open(file.replace("mask", "elevation"))
            input_data = rgb.read([1])

            mask = rasterio.open(file)
            mask_data = mask.read()

            out_data = mask_data.astype(np.uint8)
            out_data = np.where(out_data <= 0, 0, out_data)
            out_data = np.where(out_data > 0, 1, out_data)

            input_data = np.reshape(input_data, (544, 544, 1))

            val_X[idx] = input_data
            val_Y[idx] = np.reshape(out_data, (544, 544, 1))

        checkpoint = ModelCheckpoint(
            self.filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        checkpoint_loss = ModelCheckpoint(
            self.filepath_loss, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        ranges = [0,1000,2000,3000]
        for r in ranges:
            # 4160
            print("Training range : ", r)
            train_X = np.empty((1000, 544, 544, 1))
            train_Y = np.empty((1000, 544, 544, 1))

            # LOAD DATA
            file_list = glob.glob(os.path.join(
                DATA_DIR, TRAIN_DIR, "*_mask.tif"))
            file_list.sort(reverse=True)
            for idx, file in enumerate(file_list[r:r+1000]):
                rgb = rasterio.open(file.replace("mask", "elevation"))
                input_data = rgb.read([1])

                mask = rasterio.open(file)
                mask_data = mask.read()

                out_data = mask_data.astype(np.uint8)
                out_data = np.where(out_data <= 0, 0, out_data)
                out_data = np.where(out_data > 0, 1, out_data)

                input_data = np.reshape(input_data, (544, 544, 1))
                train_X[idx] = input_data
                train_Y[idx] = np.reshape(out_data, (544, 544, 1))


            tensorboard_callback = TensorBoard(
                log_dir="./logs")
            training_history = self.model.fit(train_X, train_Y, callbacks=[checkpoint, checkpoint_loss, tensorboard_callback], batch_size=1,
                                              epochs=5, validation_data=(val_X, val_Y))
            print("Average test loss: ", np.average(
                training_history.history['loss']))

        # save model
        self.model.save(os.path.join(MODEL_DIR, "bn_net_leaky.h5"))

    def load_weights(self, weights_file):
        try:
            self.model.load_weights(weights_file)
            print(f"Loaded {weights_file}")

        except Exception as e:
            print(
                f"Failed loading {os.path.join(CHECK_POINT_PATH,weights_file)}")
            print(f"Error: {e}")
            pass

    def test(self, weights):
        file_list = glob.glob("./data/train_val/test/*_mask.tif")
        test_X = np.empty((1000, 544, 544, 1))
        test_Y = np.empty((1000, 544, 544, 1))

        for idx, file in enumerate(file_list[0:1000]):
            rgb = rasterio.open(file.replace("mask", "elevation"))
            input_data = rgb.read([1])

            mask = rasterio.open(file)
            mask_data = mask.read()

            out_data = mask_data.astype(np.uint8)
            out_data = np.where(out_data <= 0, 0, out_data)
            out_data = np.where(out_data > 0, 1, out_data)
            input_data = np.reshape(input_data, (544, 544, 1))

            test_X[idx] = input_data
            test_Y[idx] = np.reshape(out_data, (544, 544, 1))
        for weights_file in weights:
            self.load_weights(weights_file)
            print(f"{weights_file}")
            scores = self.model.evaluate(test_X, test_Y, verbose=1)
            for i, metric in enumerate(self.model.metrics_names):
                print("%s: %.2f%%" % (metric, scores[i]*100))

    def predict(self, predict_X, predict_Y, plot=True, file_list = []):
        result = []

        out = self.model.predict(predict_X)
        for idx, prediction in enumerate(out):
            classes = np.where(prediction < 0.51, 0, 1)

            classes = np.reshape(classes, (544, 544))
            img = np.reshape(predict_X[idx], (544, 544))
            truth = np.reshape(predict_Y[idx], (544, 544))
            if(plot):
                pyplot.imshow(img, cmap='pink')
                pyplot.title(f'Raster elevation \n {file_list[idx]}')
                pyplot.show()
                pyplot.imshow(truth, cmap='pink')
                pyplot.title('Truth elevation')
                pyplot.show()
                pyplot.imshow(classes, cmap='pink')
                pyplot.title(f'Predited elevation, {file_list[idx]}')
                pyplot.show()
                fn = np.copy(classes)
                for x in range(544):
                    for j in range(544):
                        if(classes[x][j] !=truth[x][j] and classes[x][j]==1 ):
                            fn[x][j] = 2
                        elif(classes[x][j] !=truth[x][j] and classes[x][j]==0):
                            fn[x][j] = 3
                        elif(classes[x][j] ==truth[x][j] and classes[x][j]==1):
                            fn[x][j] = 1
                pyplot.imshow(fn, vmin=-1, vmax=len(cmap.colors), cmap=cmap)
                pyplot.show()

            result.append(classes)

        return np.array(result)

    def return_fp(self, predicted, truth):
        copy = np.copy(predicted)
        for x in range(544):
            for y in range(544):
                if(predicted[x][y] != truth[x][y]):
                    if(predicted[x][y]==1):
                        copy[x][y] = True
                    else:
                        copy[x][y] = False
                else:
                    copy[x][y] = False
        return copy

    def return_fn(self, predicted, truth):
            copy = np.copy(predicted)
            for x in range(544):
                for y in range(544):
                    if(predicted[x][y] != truth[x][y]):
                        if(predicted[x][y]==0):
                            copy[x][y] = True
                        else:
                            copy[x][y] = False
                    else:
                        copy[x][y] = False
            return copy
def plot_predictions():
    bn_net_model = BN_NET()
    bn_net_model.load_weights(WEIGHT_FILE)

    file_list = glob.glob("./data/train_val/test/*_mask.tif")
    test_X = np.empty((50, 544, 544, 1))
    test_Y = np.empty((50, 544, 544, 1))

    for idx, file in enumerate(file_list[0:50]):
        rgb = rasterio.open(file.replace("mask", "elevation"))
        input_data = rgb.read([1])

        mask = rasterio.open(file)
        mask_data = mask.read()

        out_data = mask_data.astype(np.uint8)
        out_data = np.where(out_data <= 0, 0, out_data)
        out_data = np.where(out_data > 0, 1, out_data)
        input_data = np.reshape(input_data, (544, 544, 1))

        test_X[idx] = input_data
        test_Y[idx] = np.reshape(out_data, (544, 544, 1))

    bn_net_model.predict(test_X, test_Y,file_list=file_list)


def train_model():
    bn_net_model = BN_NET()
    bn_net_model.load_weights(WEIGHT_FILE)
    bn_net_model.train()


def evaluate_model_weights():

    weights = glob.glob(os.path.join(CHECK_POINT_PATH, "*.hdf5"))
    weights.sort()
    model = BN_NET()
    model.test(weights)


if __name__ == '__main__':
    if(sys.argv[0]=='evaluate'):
        evaluate_model_weights()
    elif(sys.argv[0]=='plot'):
        plot_predictions()
    else:
        train_model()

