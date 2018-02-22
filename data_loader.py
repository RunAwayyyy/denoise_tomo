import glob
import os
import sys
import dicom
import numpy as np
from scipy.misc import imresize
from sklearn.model_selection import train_test_split

def load_data(size):
    dataset = sorted(glob.glob("./data/CPTAC-CM/DOI/C3L-00629/**/*.dcm", recursive=True))

    x_data = []
    for file in dataset:
        ds = dicom.read_file(file)
        shape = (int(ds.Rows), int(ds.Columns))
        spacing = (float(ds.SliceThickness), float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
        # if shape == (512, 512):
        array = np.zeros(shape, dtype=ds.pixel_array.dtype)
        array[:, :] = ds.pixel_array / 255.
        x_data.append(imresize(array, (size, size, 1)))
    return x_data


def load(size, noise_factor=0.2):
    if os.path.exists("./data/CPTAC-CM/data{}x{}.npy".format(size, size)):
        x_data = np.load("./data/CPTAC-CM/data{}x{}.npy".format(size, size))
    else:
        print("Load data")
        x_data = load_data(size)

        np.save("./data/CPTAC-CM/data{}x{}.npy".format(size, size), x_data)

    if x_data.shape[0] == 0:
        print("No data found, exeting!!")
        sys.exit(0)

    x_data = np.asarray(x_data).astype('float32') / 255.

    y_train, y_test = train_test_split(x_data, test_size=0.25)
    y_train = np.reshape(y_train, (len(y_train), size, size, 1))
    y_test = np.reshape(y_test, (len(y_test), size, size, 1))
    print("Training data:", y_train.shape)
    print("Test data:", y_test.shape)

    x_train_noisy = y_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=y_train.shape)
    x_test_noisy = y_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=y_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_train_noisy, x_test_noisy, y_train, y_test