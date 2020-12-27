# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:46:58 2020

"""

from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from IPython import get_ipython
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras import backend as k
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
import itertools
import numpy as np
from glob import glob
import h5py
import matplotlib.pyplot as plt

ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
    
train_path='dataset/train'
test_path='dataset/test'
valid_path='dataset/val'

nb_train_samples=3514 #60%of dataset
nb_valid_samples=1172 #20% of dataset
nb_test_samples=1170 #20% of dataset

img_width,img_height=224,224 #100,110
epochs=100
batch_size=32

if k.image_data_format()=='channels_first':
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)


optimize = ImageDataGenerator(dtype='float32', preprocessing_function=preprocess_input)
optimize_test = ImageDataGenerator(dtype='float32', preprocessing_function=preprocess_input)

train_batches=optimize.flow_from_directory(train_path,target_size=(img_width,img_height),classes=['Normal','Pneumonia'],batch_size=batch_size,shuffle=True)
valid_batches=optimize.flow_from_directory(valid_path,target_size=(img_width,img_height),classes=['Normal','Pneumonia'],batch_size=12,shuffle=True)
test_batches=optimize_test.flow_from_directory(test_path,target_size=(img_width,img_height),classes=['Normal','Pneumonia'],batch_size=nb_test_samples,shuffle=False)


# re-size all the images to this
IMAGE_SIZE = [224, 224]

#Without Preprocessing
train_path = 'dataset/train'
valid_path = 'dataset/val'

#With Clahe Preprocessing 
#train_path = 'datasetC/train'
#valid_path = 'datasetC/val'

# add preprocessing layer to the front of VGG
vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg16.layers:
  layer.trainable = False
  

  
#useful for getting number of classes
folders = glob('dataset/train/*')
#folders = glob('datasetC/train/*')

# our layers - you can add more if you want
x = Flatten()(vgg16.output)
prediction = Dense(2, activation='softmax')(x)

# create a model object
model = Model(inputs=vgg16.input, outputs=prediction)
model.summary()

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy'])

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
