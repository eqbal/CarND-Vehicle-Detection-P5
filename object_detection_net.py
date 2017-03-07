
# -*- coding: utf-8 -*-
from __future__ import print_function
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

class ObjectDetectionNet(object):

    FILE_PATH = 'model.h5'

    def __init__(self, dataset):
        self.model = None
        self.dataset = dataset
        self.input_shape =  (3,64,64)

    def build_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=self.input_shape, output_shape=self.input_shape))
        model.add(Convolution2D(10, 3, 3, activation='relu', name='conv1',input_shape=input_shape, border_mode="same"))
        model.add(Convolution2D(10, 3, 3, activation='relu', name='conv2',border_mode="same"))
        model.add(MaxPooling2D(pool_size=(8,8)))
        model.add(Dropout(0.25))
        model.add(Convolution2D(128,8,8,activation="relu",name="dense1"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(1,1,1,name="dense2", activation="tanh"))
        model.add(Flatten())

        self.model = model

    def summary(self):
        self.model.summary()

    def compile(self):
        self.model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])


    def train(self):
        print('Start training.')
        self.model.fit(
                self.X_train,
                self.Y_train,
                batch_size=128,
                nb_epoch=20,
                verbose=1,
                validation_data=(self.X_test, self.Y_test)
        )

    def save(self):
        self.model.save(self.FILE_PATH)
        print('Model Saved.')

    def load(self):
        print('Model Loaded.')
        self.model = load_model(self.FILE_PATH)

    def evaluate(self):
        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def predict(self, image):
        image = image.astype('float32')
        result = self.model.predict(image)
        print(result)
        return result[0]

