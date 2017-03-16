# -*- coding: utf-8 -*-
from __future__ import print_function
import random

from cnn_helpers import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pylab as plt

class ObjectDetectionNet(object):

    FILE_PATH = 'yolo.weights'

    def __init__(self):
        self.model = None
        self.input_shape = (3,448,448)

    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(16, 3, 3,input_shape=self.input_shape,border_mode='same',subsample=(1,1)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(64,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(128,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(256,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(512,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        model.add(Convolution2D(1024,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Convolution2D(1024,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Convolution2D(1024,3,3 ,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(1470))

    def summary(self):
        self.model.summary()

    def compile(self):
        self.model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])

    def load_weights():
        data = np.fromfile(self.FILE_PATH,np.float32)
        data=data[4:]

        index = 0
        for layer in self.model.layers:
            shape = [w.shape for w in layer.get_weights()]
            if shape != []:
                kshape,bshape = shape
                bia = data[index:index+np.prod(bshape)].reshape(bshape)
                index += np.prod(bshape)
                ker = data[index:index+np.prod(kshape)].reshape(kshape)
                index += np.prod(kshape)
                layer.set_weights([ker,bia])

    def train(self):
        print('Start training.')
        self.model.fit(
                self.dataset.X_train,
                self.dataset.Y_train,
                batch_size=128,
                nb_epoch=20,
                verbose=1,
                validation_data=(self.dataset.X_test, self.dataset.Y_test)
        )

    def evaluate(self):
        score = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def predict(self, image):
        image  = image.astype('float32')
        result = self.model.predict_proba(image)
        print(result)
        return result[0]
