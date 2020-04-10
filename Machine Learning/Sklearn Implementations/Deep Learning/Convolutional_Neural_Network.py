#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:07:32 2020

@author: admangli
"""

import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
# MAC OS specific OpenMP fix
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

target_image_height = 128
target_image_width = 128
batch_size = 64

#%%
# Network design
cnn = Sequential()
# Shape is (256 256 x 3)
cnn.add(Convolution2D(filters = 64, kernel_size = (3, 3), activation = 'relu', 
                      input_shape=(target_image_height, target_image_width, 3), data_format = 'channels_last'))
cnn.add(BatchNormalization())# Shape is (254 x 254 x 32)
cnn.add(MaxPooling2D(pool_size = (2, 2)))# Shape is (127 x 127 x 32)
cnn.add(Dropout(0.2))
cnn.add(Convolution2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
cnn.add(BatchNormalization())# Shape is (125 x 125 x 64)
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Convolution2D(filters = 256, kernel_size = (3, 3), activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.3))
cnn.add(Convolution2D(filters = 512, kernel_size = (3, 3), activation = 'relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size = (2, 2)))# Shape is (63 x 63 x 64)
cnn.add(Dropout(0.3))
cnn.add(Flatten())# Shape is (254016, )
cnn.add(Dense(units = 1024, kernel_initializer = 'uniform', activation = 'relu'))
cnn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(cnn.summary())
#%%
from keras.preprocessing.image import ImageDataGenerator

# Image Augmentation - Prevents overfitting

dataset_loc = '/Users/admangli/Personal/Learning/large_datasets/dogcats/train/'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(
        dataset_loc,
        target_size=(target_image_height, target_image_width),
        batch_size=batch_size,
        class_mode='binary',
        subset = 'training')

validation_generator = train_datagen.flow_from_directory(
        dataset_loc,
        target_size=(target_image_width, target_image_width),
        batch_size=batch_size,
        class_mode='binary',
        subset = 'validation')

#This method fits the model to dataset and simultaneously tests
cnn.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps = validation_generator.samples // batch_size)
