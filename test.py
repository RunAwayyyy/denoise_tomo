import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from models import ImageSuperResolutionModel
from data_loader import load

import argparse

parser = argparse.ArgumentParser(description='Denoise CT Reconstructions')

parser.add_argument('-b', action='store', dest='batch_size', default=128, type=int, help='Size of a batch')
parser.add_argument('-s', action='store', dest='size', default=128, type=int, help='Width and height of image')
parser.add_argument('-n', action='store', dest='noise_factor', default=0.2, type=float, help='Float between 0 and 1')
parser.add_argument('-e', action='store', dest='nb_epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--load-weights', dest='load_weights', action='store_true')
parser.add_argument('--no-load-weights', dest='load_weights', action='store_false')
parser.set_defaults(feature=True)

args = parser.parse_args()
batch_size = args.batch_size
size = args.size
noise_factor = args.noise_factor
nb_epochs = args.nb_epochs
load_weights = args.load_weights

x_train_noisy, x_test_noisy, y_train, y_test = load(size, noise_factor)

datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True)

datagen.fit(x_train_noisy)

n = 5
plt.figure()
indices = np.random.randint(0, len(x_test_noisy), size=n)
for i, ind in enumerate(indices):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_train_noisy[ind].reshape(size, size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

autoencoder = ImageSuperResolutionModel()
autoencoder.create_model(height=size, width=size, channels=1, load_weights=load_weights)

autoencoder.model.summary()

autoencoder.fit(datagen, x_train_noisy, y_train, batch_size=batch_size, nb_epochs=nb_epochs, save_history=True)

decoded_imgs = autoencoder.predict(y_test, batch_size=batch_size)

n = 5
plt.figure()
indices = np.random.randint(0, len(x_test_noisy), size=n)
for i, ind in enumerate(indices):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[ind].reshape(size, size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[ind].reshape(size, size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
