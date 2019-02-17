#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8

from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys

import os, shutil                         # For issuing commands to the OS.

import numpy as np
import time
import random
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy import stats
import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import datetime
from scipy.optimize import curve_fit
from scipy.stats import bernoulli
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import brier_score_loss

# Instantiating a small convnet for dogs vs. cats classification
from keras.utils import np_utils
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16 
from keras.applications.inception_v3 import InceptionV3
from PIL import Image
import create_folders # the class that creates working folders with test, train and validation images

# can import other image-classification models(all pretrained on the ImageNet dataset)                                     
# possible models are:
# Xception
# Inception V3 
# ResNet50
# VGG16
# VGG19
# MobileNet


#-----------------------------------------------------------------------------------------
# FEATURE EXTRACTION WITHOUT DATA AUGMENTATION (following Chollet)
#-----------------------------------------------------------------------------------------

#----------
# In the current classification problem there are 8 categories  
#----------
categories = ['01_TUMOR','02_STROMA','03_COMPLEX','04_LYMPHO','05_DEBRIS','06_MUCOSA','07_ADIPOSE','08_EMPTY']
pix = 150                        # number of pixels
nb_classes = len(categories)

# each subdirectory has a total number of 625 files
total_nb = 625
nb_train = 425
nb_test = 100
nb_valid = 100

batch_size = 25                  # can be chosen for the training


# ----------------------------------------------------------------------------------------
# I. Create WORKING DIRECTORIES (using the class CreateFolders)
#-----------------------------------------------------------------------------------------


# if the working folders have to be created then call the class CreateFolders and the function get_folders
folders_already_exist = True

obj = create_folders.CreateFolders(total_nb,nb_train,nb_test,nb_valid,categories) # file_name.class_name

if folders_already_exist == False:
   train_dir,validation_dir,test_dir = obj.get_folders()                             # obj.function in class
else:
   train_dir,validation_dir,test_dir = obj.directories() 

    
     
# ----------------------------------------------------------------------------------------
# II. Instantiating the VGG176 model (for the use of a pre-trained model)
#-----------------------------------------------------------------------------------------
 
# 
# weights ......specifies the weight checkpoint from which to initialize the model.
# include_top ..refers to including (or not) the densely connected classifier on top of the
# network. By default, this densely connected classifier corresponds to the 1,000 classes 
# from ImageNet. Because you intend to use your own densely connected classifier 
# (with only two classes: cat and dog), you don’t need to include it.
# input_shape... is the shape of the image tensors that you’ll feed to the network.
# This argument is purely optional: if you don’t pass it, the network will be able to 
# process inputs of any size.
# dimension1 in the last layer is 3 if Inception V3, 4 if VGG16
# dimension2                     2048 if Inception V3, 512 if VGG16


dim1 = 4   # 3
dim2 = 512 #2048




#modl = 'Inception_v3'
modl = 'VGG16'
if modl == 'VGG16':
   conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(pix, pix, 3))

   #conv_base = InceptionV3(weights='imagenet',
   #                  include_top=False,
   #                  input_shape=(pix, pix, 3))

   conv_base.summary()






#-----------------------------------------------------------------------------------------
# III. FAST FEATURE EXTRACTION WITHOUT DATA AUGMENTATION
#      Extracting features using the pre-trained convolutional base
#-----------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------
# 1. Data preprocessing  (done with ImageDataGenerator) and running the data through the
# pretrained CNN
#-----------------------------------------------------------------------------------------

# Data should be formatted into appropriately preprocessed floatingpoint tensors before 
# being fed into the network. Currently, the data sits on a drive as JPEG files, so the 
# steps for getting it into the network are roughly as follows:
# 1 Read the picture files.
# 2 Decode the JPEG content to RGB grids of pixels.
# 3 Convert these into floating-point tensors.
# 4 Rescale the pixel values (between 0 and 255) to the [0, 1] interval 
#   (neural networks prefer to deal with small input values)


# The class ImageDataGenerator quickly sets up Python generators that can automatically 
# turn image files on disk into batches of preprocessed tensors.

# flow from directory: Takes the path to a directory & generates batches of augmented data.
# class_mode: "categorical" will be 2D one-hot encoded labels (yes!!! automatical)
# classes: Optional list of class subdirectories (e.g. ['dogs', 'cats']). Default: None. 
# If not provided, the list of classes will be automatically inferred from the subdirectory 
# names/structure under directory, where each subdirectory will be treated as a different 
# class (and the order of the classes, which will map to the label indices, will be alphanumeric).  
# The dictionary containing the mapping from class names to class indices can be obtained 
# via the attribute class_indices

datagen = ImageDataGenerator(rescale=1./255) 

def extract_features(directory, sample_count):

    features = np.zeros(shape=(sample_count, dim1, dim1, dim2)) 
    labels = np.zeros(shape=(sample_count,nb_classes))
     

    #--------------------------------
    # read in images from directories
    #--------------------------------

    # flow_from_directory: Takes the path to a directory & generates batches of augmented (???or not?) data.

    # because generators yield data indefinitely in a loop, 
    # you must break after every image has been seen once.



    generator = datagen.flow_from_directory(
                                       directory, 
                                       target_size = (pix, pix), 
                                       batch_size = batch_size, 
                                       class_mode='categorical')
    i=0
      
    for inputs_batch, labels_batch in generator:

        # fetch the pre-trained data and corresponding labels

        #print('inputs_batch',inputs_batch.shape)
        #print('labels_batch',labels_batch.shape)

        features_batch = conv_base.predict(inputs_batch)
        
        features[i * batch_size : (i + 1) * batch_size] = features_batch 
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

#data batch shape: (25, 150, 150, 3)
#labels batch shape: (25,)

print('extract predictions from model base')

train_features, train_labels = extract_features(train_dir, nb_train)
validation_features, validation_labels = extract_features(validation_dir, nb_valid) 
test_features, test_labels = extract_features(test_dir, nb_test)

print('train features',train_features.shape)
print('validation features',validation_features.shape)
print('test features',test_features.shape)
#print('reshape the features')
# 512 is the last dimension of MaxPooling layer at the end of the CNN... 

# reshape the training, validation, test data that were pretrained through the CNN
train_features = np.reshape(train_features, (nb_train, dim1 * dim1 * dim2))
validation_features = np.reshape(validation_features, (nb_valid, dim1 * dim1 * dim2))
test_features = np.reshape(test_features, (nb_test, dim1 * dim1 * dim2))

#---
# define and train the densely connected classifier
# then plot

print('next: Now we will define and train the densely connected classifier')
#exit()
#-----------------------------------------------------------------------------------------
# 2. Stick a densely connected classifier
#
# run a conv base once for every input image and use the resulting output as input to a DC 
# classifier
# Extract features using the pretrained convolutional base
# Define and train the densely connected classifier (dropout regularization) and train it 
# on the labeled data
#-----------------------------------------------------------------------------------------

# Recall:
# dimension1 in the last layer is 3 if Inception V3, 4 if VGG16
# dimension2                     2048 if Inception V3, 512 if VGG16

# changeable (hyper-)parameters:
nb_epochs = 300
nodes_in_lastlayer1 = 60
nodes_in_lastlayer2 = 0 #20 #500 # 300
dropout1 = 0.55
dropout2 = 0.5

model = models.Sequential()
model.add(layers.Dense(nodes_in_lastlayer1, activation='relu', input_dim=dim1 * dim1 * dim2))
model.add(layers.Dropout(dropout1))


#model.add(layers.Dense(nodes_in_lastlayer2, activation='relu', input_dim=dim1 * dim1 * dim2))
#model.add(layers.Dropout(dropout2))

#model.add(layers.Dense(nodes_in_lastlayer2, activation='relu', input_dim=dim1 * dim1 * dim2))
#model.add(layers.Dropout(dropout2))





model.add(layers.Dense(8, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), # optimizer could be 'adam'
              loss='categorical_crossentropy',            # loss could be 'categorical_crossentropy'
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=nb_epochs,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))


# plot results

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure()
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.savefig('Accuracy_'+str(modl)+'_nodesL1_'+str(nodes_in_lastlayer1)+'_nodesL2_'+str(nodes_in_lastlayer2)+'_epochs'+str(nb_epochs)+'_dropout'+str(dropout1)+'.png')





plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Loss'+str(modl)+'_nodesL1_'+str(nodes_in_lastlayer1)+'_nodesL2_'+str(nodes_in_lastlayer2)+'_epochs'+str(nb_epochs)+'_dropout'+str(dropout1)+'.png')





