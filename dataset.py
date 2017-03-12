import glob
import cv2
import numpy as np
import skimage
from skimage import data, color, exposure
from sklearn.model_selection import train_test_split

class Dataset(object):

    def __init__(self):
        self.cars     = glob.glob("../data/vehicles/*/*.png")
        self.non_cars = glob.glob("../data/non-vehicles/*/*.png")
        self.X        = []

    def call(self):
        self.generate_Y_vector()
        self.generate_X_vector()
        self.validation_split()

    def generate_Y_vector(self):
        self.Y = np.concatenate([
            np.ones(len(self.cars)),
            np.zeros(len(self.non_cars))-1
        ])

    def generate_X_vector(self):
        for name in self.cars:
            self.X.append(skimage.io.imread(name))
        for name in self.non_cars:
            self.X.append(skimage.io.imread(name))
        self.X = np.array(self.X)

    def validation_split(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.20, random_state = np.random.randint(0, 100)
        )
        self.X_train = self.X_train.astype('float32')
        self.X_test  = self.X_test.astype('float32')

    def inspect(self):
        print('X_train shape:', self.X_train.shape)
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')
        print(len(self.cars), 'images of vehicles')
        print(len(self.non_cars), 'images of non vehicles')
