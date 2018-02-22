import numpy as np
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D
from keras import backend as K
import keras.callbacks as callbacks
import keras.optimizers as optimizers
from keras.callbacks import ModelCheckpoint
from advanced import HistoryCheckpoint, TensorBoardBatch

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))

class ImageSuperResolutionModel:
    def __init__(self):

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights_isrm.h5"
        self.model_name = "ImageSuperResolutionModel"

    def create_model(self, height=32, width=32, channels=3, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        shape = (width, height, channels)
        init = Input(shape=shape)
        x = Conv2D(self.n1, (self.f1, self.f1), activation='relu', padding='same', name='level1')(init)
        x = Conv2D(self.n2, (self.f2, self.f2), activation='relu', padding='same', name='level2')(x)

        out = Conv2D(channels, (self.f3, self.f3), padding='same', name='output')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)
        self.model = model

    def fit(self, generator, x_train, y_train, batch_size=128, nb_epochs=100, save_history=True, history_fn="Model_History.txt"):
        """
        Standard method to train any of the models.
        """
        callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                                   mode='max', save_weights_only=True, verbose=2)]
        if save_history:
            callback_list.append(HistoryCheckpoint(history_fn))

            if K.backend() == 'tensorflow':
                log_dir = './%s_logs/' % self.model_name
                tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
                callback_list.append(tensorboard)

        print("Training model : %s" % (self.__class__.__name__))
        self.model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size),
                                 steps_per_epoch=len(x_train) // batch_size,
                                 epochs=nb_epochs, callbacks=callback_list)

    def predict(self, x_test, batch_size=128):
        return self.model.predict(x_test, batch_size=batch_size)


class DeConvolv:
    def __init__(self):
        self.weight_path = "deblur_cnn_weights.h5.h5"
        self.model_name = "ImageSuperResolutionModel"

    def __conv_batch(self, input, filter, kernel):
        x = Conv2D(filters=filter, kernel_size=kernel, strides=1, padding='same')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def create_model(self, height=32, width=32, channels=3, load_weights=False):
        input = Input(shape=(height, width, channels))

        # HIDDEN LAYERS
        x = self.__conv_batch(input, filter=128, kernel=10)
        x = self.__conv_batch(x, filter=320, kernel=1)
        x = self.__conv_batch(x, filter=320, kernel=1)
        x = self.__conv_batch(x, filter=320, kernel=1)
        x = self.__conv_batch(x, filter=128, kernel=1)
        x = self.__conv_batch(x, filter=128, kernel=3)
        x = self.__conv_batch(x, filter=512, kernel=1)
        x = self.__conv_batch(x, filter=128, kernel=5)
        x = self.__conv_batch(x, filter=128, kernel=5)
        x = self.__conv_batch(x, filter=128, kernel=3)
        x = self.__conv_batch(x, filter=128, kernel=5)
        x = self.__conv_batch(x, filter=128, kernel=5)
        x = self.__conv_batch(x, filter=256, kernel=1)
        x = self.__conv_batch(x, filter=64, kernel=7)

        x = Conv2D(filters=channels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     , kernel_size=7, strides=1, padding='same', activation='relu')(x)

        adam = optimizers.Adam(lr=1e-3)
        model = Model(inputs=input, outputs=x)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)
        self.model = model

    def fit(self, generator, x_train, y_train, batch_size=128, nb_epochs=100, save_history=True, history_fn="Model_History.txt"):
        """
        Standard method to train any of the models.
        """
        callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                                   mode='max', save_weights_only=True, verbose=2)]
        if save_history:
            callback_list.append(HistoryCheckpoint(history_fn))

            if K.backend() == 'tensorflow':
                log_dir = './%s_logs/' % self.model_name
                tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
                callback_list.append(tensorboard)

        print("Training model : %s" % (self.__class__.__name__))
        self.model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size),
                                 steps_per_epoch=len(x_train) // batch_size,
                                 epochs=nb_epochs, callbacks=callback_list)

    def predict(self, x_test, batch_size=128):
        return self.model.predict(x_test, batch_size=batch_size)