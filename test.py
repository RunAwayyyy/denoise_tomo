import dicom
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.misc import imresize

# Keras imports
from keras.models import Model
from keras import backend as K


def load_data():
    dataset = sorted(glob.glob("./data/CPTAC-CM/DOI/C3L-00629/**/*.dcm", recursive=True))

    x_data = []
    for file in dataset:
        ds = dicom.read_file(file)
        shape = (int(ds.Rows), int(ds.Columns))
        spacing = (float(ds.SliceThickness), float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
        # if shape == (512, 512):
        array = np.zeros(shape, dtype=ds.pixel_array.dtype)
        array[:, :] = ds.pixel_array / 255.
        x_data.append(imresize(array, (128, 128, 1)))
    return x_data

if False and os.path.exists("./data/CPTAC-CM/data.npy"):
    x_data = np.load("./data/CPTAC-CM/data.npy")
else:
    print("Load data")
    x_data = load_data()

    np.save("./data/CPTAC-CM/data.npy", x_data)

x_data = np.asarray(x_data).astype('float32') / 255.

x_train, x_test = train_test_split(x_data, test_size=0.25)
x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))
x_test = np.reshape(x_test, (len(x_test), 64, 64, 1))
print("Training data:", x_train.shape)
print("Test data:", x_test.shape)

noise_factor = 0.2
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)


x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
indices = np.random.randint(0, len(x_test_noisy), size=n)
for i, ind in enumerate(indices):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_train_noisy[ind].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = Input(shape=(64, 64, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_split=0.25)

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
indices = np.random.randint(0, len(x_test_noisy), size=n)
for i, ind in enumerate(indices):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noisy[ind].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[ind].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()