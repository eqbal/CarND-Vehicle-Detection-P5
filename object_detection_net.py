# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import keras

from cnn_helpers import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pylab as plt
from keras.layers.advanced_activations import LeakyReLU

class ObjectDetectionNet(object):

    FILE_PATH = 'yolo.weights'

    def __init__(self):
        self.model = None
        keras.backend.set_image_dim_ordering('th')

    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
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

        self.model = model

    def summary(self):
        self.model.summary()

    def pipeline(self, image):
        crop = image[300:650,500:,:]
        resized = cv2.resize(crop,(448,448))
        batch = np.array([resized[:,:,0],resized[:,:,1],resized[:,:,2]])
        batch = 2*(batch/255.) - 1
        batch = np.expand_dims(batch, axis=0)
        out = self.model.predict(batch)
        boxes = out_to_car_boxes(out[0], threshold = 0.17)
        return draw_box(boxes,image,[[500,1280],[300,650]])

    def load_weights(self):
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


    def predict(self, image):
        image  = image.astype('float32')
        result = self.model.predict_proba(image)
        print(result)
        return result[0]

