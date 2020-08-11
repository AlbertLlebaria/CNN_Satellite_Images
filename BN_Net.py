from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Activation, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, GlobalMaxPooling2D, BatchNormalization, LeakyReLU, UpSampling2D
from keras.models import Sequential, load_model
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
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(
            f" — val_f1: {_val_f1} — val_precision: {_val_precision}— val_recall {_val_recall}")
        return


metrics = Metrics()

np.seterr(divide='ignore', invalid='ignore')

MODEL_DIR = './model'
DATA_DIR = './data'
TRAIN_DIR = 'train_val/train'
VAL_DIR = 'train_val/validation'
CHECK_POINT_PATH = "train_ckpt/leaky"
WEIGHT_FILE = "train_ckpt/leaky/weights-improvement-02-0.93.hdf5"
DRORATE = 0.25
LEARNING_RATE = 2*math.pow(10, -4)


class BN_NET:
    def __init__(self, drop_rate=DRORATE, learning_rate=LEARNING_RATE, model_dir=MODEL_DIR):
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.input_shape = (544, 544, 1)
        self.model_dir = model_dir
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model = self.create_model()
        self.filepath = "./train_ckpt/leaky/weights-improvement-{epoch:02d}-{val_accuracy:.3f}.hdf5"

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
        return MaxPooling2D(pool_size=2)(bn_2), drop

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

            val_X[idx] = input_data/255
            val_Y[idx] = np.reshape(out_data, (544, 544, 1))

        checkpoint = ModelCheckpoint(
            self.filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        ranges = [0, 1000, 2000, 3000]
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
                train_X[idx] = input_data/255
                train_Y[idx] = np.reshape(out_data, (544, 544, 1))

                # create image data augmentation generator
            # data augmentation
            datagen = ImageDataGenerator(rotation_range=90)
            # prepare iterator
            it = datagen.flow(train_X, train_Y, batch_size=1)
            tensorboard_callback = TensorBoard(
                log_dir="./logs")
            training_history = self.model.fit(it, callbacks=[checkpoint, tensorboard_callback, metrics],
                                              epochs=5, validation_data=(val_X, val_Y))
            print("Average test loss: ", np.average(
                training_history.history['loss']))

        # save model
        self.model.save(os.path.join(MODEL_DIR, "bn_net.h5"))

    def load_weights(self, weights_file):
        try:
            self.model.load_weights(weights_file)
            print(f"Loaded {weights_file}")

        except Exception as e:
            print(
                f"Failed loading {os.path.join(CHECK_POINT_PATH,weights_file)}")
            print(f"Error: {e}")
            pass

    def test(self, weights_file):
        self.load_weights(weights_file)
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

            test_X[idx] = input_data/255
            test_Y[idx] = np.reshape(out_data, (544, 544, 1))

        scores = self.model.evaluate(test_X, test_Y, verbose=1)
        for i, metric in enumerate(self.model.metrics_names[1]):
            print("%s: %.2f%%" % (metric, scores[i]*100))

    def predict(self, predict_X, predict_Y, plot=True):
        result = []

        out = self.model.predict(predict_X)
        for idx, prediction in enumerate(out):
            classes = np.where(prediction < 0.51, 0, 255)

            classes = np.reshape(classes, (544, 544))
            img = np.reshape(predict_X[idx], (544, 544))
            truth = np.reshape(predict_Y[idx], (544, 544))
            if(plot):
                pyplot.imshow(img, cmap='pink')
                pyplot.title('Raster elevation')
                pyplot.show()
                pyplot.imshow(truth, cmap='pink')
                pyplot.title('Truth elevation')
                pyplot.show()
                pyplot.imshow(classes, cmap='pink')
                pyplot.title('Predited elevation')
                pyplot.show()
            result.append(classes)

        return np.array(result)


def main():
    bn_net_model = BN_NET()
    bn_net_model.train()
    # bn_net_model.load_weights(
    #     '/Volumes/TOSHIBA EXT/learning/CNN_Satellite_Images/train_ckpt/leaky/weights-improvement-03-0.851.hdf5')
    # bn_net_model.train()

    weights = glob.glob(os.path.join(CHECK_POINT_PATH, "*.hdf5"))
    weights.sort()
    model = BN_NET()
    for weight_file in weights:
        model.test(weight_file)


if __name__ == '__main__':
    main()
