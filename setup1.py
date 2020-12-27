# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:48:50 2020

"""
from keras.models import load_model
from keras import optimizers
import tensorflow as tf
import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten,Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import *
from keras.layers.convolutional import Conv2D,MaxPooling2D
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import classification_report
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#Without Preprocesssing
train_path='dataset/train'
test_path='dataset/test'
valid_path='dataset/val'

'''
#Preprocessed with Clahe
train_path='datasetC/train'
test_path='datasetC/test'
valid_path='datasetC/val'
'''

nb_train_samples=3514 #60%of dataset
nb_valid_samples=1172 #20% of dataset
nb_test_samples=1170 #20% of dataset

img_width,img_height=150,150
epochs=100
batch_size=32

if k.image_data_format()=='channels_first':
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)

optimize=ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        fill_mode='nearest',
        shear_range=0.2,
        zoom_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        rescale=1./255)

optimize_test=ImageDataGenerator(
        rescale=1./255)

train_batches=optimize.flow_from_directory(train_path,target_size=(img_width,img_height),classes=['Normal','Pneumonia'],batch_size=batch_size,shuffle=True)
valid_batches=optimize.flow_from_directory(valid_path,target_size=(img_width,img_height),classes=['Normal','Pneumonia'],batch_size=12,shuffle=True)
test_batches=optimize_test.flow_from_directory(test_path,target_size=((img_width,img_height),classes=['Normal','Pneumonia'],batch_size=1170,shuffle=False)



#create model
model = Sequential()#add model layers

model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(img_width,img_height,3)))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.summary() 
 
adam = optimizers.Adam(lr=0.001)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
checkpoint= ModelCheckpoint("model.h5",monitor='val_accuracy',verbose=1,save_best_only=True,save_weights_only=False,mode='auto',period=1)
early=EarlyStopping(monitor='val_accuracy', min_delta=0,patience=5,verbose=1,mode='auto')
hist=model.fit_generator(train_batches,steps_per_epoch=nb_train_samples//batch_size,
                    validation_data=valid_batches,validation_steps=nb_valid_samples//12,epochs=epochs,callbacks=[checkpoint,early])


print(hist.history.keys())
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()