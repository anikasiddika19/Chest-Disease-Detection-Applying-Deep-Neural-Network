# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:35:57 2020

"""
from keras import backend as k
from keras.models import Model
from keras.models import load_model
import numpy as np
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
    
#Without Preprocesssing
test_path='dataset/test'

#Preprocessed with Clahe
#test_path='datasetC/test'


nb_test_samples=1170

img_width,img_height=150,150

#VGG16
#img_width,img_height=224,224

if k.image_data_format()=='channels_first':
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)

optimize_test=ImageDataGenerator(
        rescale=1./255)
test_batches=optimize_test.flow_from_directory(test_path,target_size=(img_width,img_height),classes=['Normal','Pneumonia'],batch_size=nb_test_samples,shuffle=False)

model=load_model('model.h5')
model.summary()

test_imgs,test_labels=next(test_batches)
test_labels = np.argmax(test_labels, axis=1) 

predictions=model.predict_generator(test_batches,steps=1,verbose=0)
predictions = np.argmax(predictions, axis=1)

'''test_batches.class_indices'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm = confusion_matrix(test_labels,predictions)

cm_plot_labels=['Normal','Pneumonia']
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')

target_names = ['Normal','Pneumonia']
print(classification_report(test_labels,predictions, target_names=target_names))
