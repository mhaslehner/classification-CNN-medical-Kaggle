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
from keras.applications import VGG16 # can import other image-classification models 
                                     # (all pretrained on the ImageNet dataset)

from PIL import Image

# possible models are:
# Xception
# Inception V3 
# ResNet50
# VGG16
# VGG19
# MobileNet

#---------------------
# how to read in tif images
#read image

print('reading in tif images')

# folder
# /Users/user/Documents/workspace/MedicalML/project2/colorectal-histology-mnist

img_arr = plt.imread("colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/01_TUMOR/1A11_CRC-Prim-HE-07_022.tif_Row_601_Col_151.tif")
#view image
plt.imshow(img_arr)
# plt.show()
imarray = np.array(img_arr)
# check the size of the image as a np array 
# we have an image of 150x150 pixels
print('shape of image array',imarray.shape)
# one can turn the array back into a PIL image like this:
Image.fromarray(imarray)
#exit()



#-----------------------------------------------------------------------------------------
# FEATURE EXTRACTION WITHOUT DATA AUGMENTATION
#-----------------------------------------------------------------------------------------



# ------------------------------------------------
# I. Create WORKING DIRECTORIES
#-------------------------------------------------


#-----------------------------------------------------------------------------------------     
# 1. initial path  ----   
# colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/01_TUMOR/,....,/08_EMPTY

# 2. work path    
# this creates us the folders: mainpath/train/                01_TUMOR/,...,08_EMPTY/
#                                      /validation/           01_TUMOR/,...,08_EMPTY/        
#                                      /test/                 01_TUMOR,/...08_EMPTY/ 
#-----------------------------------------------------------------------------------------                                           
 
# linux command for counting nb of lines in each directory
# for d in *; do ls $d | wc -l; done
# every category in the directory has 625 images


# create train, test, valid directories
base_dir = '/Users/user/Documents/workspace/MedicalML/project2/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000'
# here we have 8 categories for the classification problem
categories = ['01_TUMOR','02_STROMA','03_COMPLEX','04_LYMPHO','05_DEBRIS','06_MUCOSA','07_ADIPOSE','08_EMPTY']


# each subdirectory has a total number of 625 
total_nb = 625
nb_train = 425
nb_test = 100
nb_valid = 100
batch_size = 25 # choose a 
pix = 150 # pixels
nb_classes = len(categories)

TRAIN = 'train'
TEST = 'test'
VALID = 'validation'

work_folder = [TRAIN,TEST,VALID]
work_ranges = [range(0,nb_train),range(nb_train,nb_train + nb_test),range(nb_train + nb_test,total_nb)]
 

def init_path(ind_of_cat): 
    # creates path to initial files for given category ind_of_cat
     
    # name of category(ind_of_cat)
    c = categories[ind_of_cat]  
    path = base_dir + '/' + c
    return path

def work_path(ind_of_cat,dir): 
   # creates pathnames to work files. The work files will subsequently be copied to this path name     
   # dir is one of train, test or valid 

   c = categories[ind_of_cat]  
   path = base_dir + '/' + dir + '/' + c

   return path

# 1. create directory of the type base_dir/train,val,test
# into these directories we will then copy the files that are contained in each of the subdirectories?

train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

# and each category has to be split into train, test and dev sets

# 2. create subdirectories for the different classes
#    base_dir/train/cat1,....,cat8
#    base_dir/test/cat1,....,cat8
#    base_dir/valid/cat1,....,cat8

for dir in [train_dir, validation_dir, test_dir]: 
   for name in categories: # classes
        # create directory of the type # path/train/        
        subdir = os.path.join(dir,name)        
#        os.mkdir(subdir)



#----------------------------------------------
# copy files from initial to work folders 
#----------------------------------------------
for i in range(len(categories)): # classes    

    # list names of files contained in the initial path    
    files_in_init = os.listdir(init_path(i)) 

    # copy a certain nb of files to train working directory        
    for f in range(3):                        # each folder train,test,valid   
      for j in work_ranges[f]:                # nb of images in train, test, valid folders
         
         src = init_path(i) + '/' + files_in_init[j]
         dst = work_path(i,work_folder[f]) + '/' + files_in_init[j] # path + filename
         #shutil.copyfile(src, dst)
       


print('created working directory base_dir/train,test,valid/01_TUMOR/,...,08_EMPTY/')
# As a sanity check, let’s count how many pictures are in each training split (train/vali- dation/test):
# >>> print('total training cat images:', len(os.listdir(train_cats_dir))) total training cat images: 1000



     
# ----------------------------------------------------------
# II. Instantiating the VGG176 model (for the use of a pre-trained model)
#-----------------------------------------------------------
 
# 
# weights specifies the weight checkpoint from which to initialize the model.
# include_top refers to including (or not) the densely connected classifier on top of the
# network. By default, this densely connected classifier corresponds to the 1,000 classes 
# from ImageNet. Because you intend to use your own densely connected classifier 
# (with only two classes: cat and dog), you don’t need to include it.

# input_shape is the shape of the image tensors that you’ll feed to the network.
# This argument is purely optional: if you don’t pass it, the network will be able to 
# process inputs of any size.

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(pix, pix, 3))

conv_base.summary()



#-------------------------------
# III. FAST FEATURE EXTRACTION WITHOUT DATA AUGMENTATION
#      Extracting features using the pre-trained convolutional base
#-----------------------------------------



# 1. Data preprocessing  (done with ImageDataGenerator) and running the data through the
# pretrained CNN

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

    features = np.zeros(shape=(sample_count, 4, 4, 512)) 
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

        print('inputs_batch',inputs_batch.shape)
        print('labels_batch',labels_batch.shape)    
        #print('labels_batch',labels_batch)   
        #exit()    
        features_batch = conv_base.predict(inputs_batch)
        
        features[i * batch_size : (i + 1) * batch_size] = features_batch 
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

#>>> for data_batch, labels_batch in train_generator:
#>>>     print('data batch shape:', data_batch.shape)
#>>>     print('labels batch shape:', labels_batch.shape)
#>>>     break
#data batch shape: (20, 150, 150, 3)
#labels batch shape: (20,)



train_features, train_labels = extract_features(train_dir, nb_train) 
validation_features, validation_labels = extract_features(validation_dir, nb_valid) 
test_features, test_labels = extract_features(test_dir, nb_test)

print('train features',train_features.shape)
print('validation features',validation_features.shape)
print('test features',test_features.shape)
print('reshape the features')
# 512 is the last dimension of MaxPooling layer at the end of the CNN... 
# reshape the training, validation, test data that were pretrained through the CNN
train_features = np.reshape(train_features, (nb_train, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (nb_valid, 4 * 4 * 512))
test_features = np.reshape(test_features, (nb_test, 4 * 4 * 512))
#---
# define and train the densely connected classifier
# then plot

print('next: Now we will define and train the densely connected classifier')

#exit()

#---------------------------------------
# 2. Stick a densely connected classifier
#
# run a conv base once for every input image and use the resulting output as input to a DC 
# classifier
# Extract features using the pretrained convolutional base
# Define and train the densely connected classifier (dropout regularization) and train it 
# on the labeled data
#-----------------------------------------------------------------------------------------


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), # optimizer could be 'adam'
              loss='categorical_crossentropy',            # loss could be 'categorical_crossentropy'
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=25,
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
plt.savefig('firstCNNofMylene_accuracy.png')




plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('firstCNNofMylene_loss.png')

#plt.show()



#-------------------
# in github we have
#-------------------

# batch_size = 128
# nb_epoch = 10
# data_augmentation = True
# 
# # Model saving callback
# #checkpointer = ModelCheckpoint(filepath='stochastic_depth_cifar10.hdf5', verbose=1, save_best_only=True)
# 
# if not data_augmentation:
#     print('Not using data augmentation.')
#     history = model.fit(x_train, y_train, 
#                         batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
#                         validation_data=(x_test, y_test), shuffle=True,
#                         callbacks=[])
# else:
#     print('Using real-time data augmentation.')
# 
#     # realtime data augmentation
#     datagen_train = ImageDataGenerator(
#         featurewise_center=False,
#         samplewise_center=False,
#         featurewise_std_normalization=False,
#         samplewise_std_normalization=False,
#         zca_whitening=False,
#         rotation_range=0,
#         width_shift_range=0.125,
#         height_shift_range=0.125,
#         horizontal_flip=True,
#         vertical_flip=False)
#     datagen_train.fit(x_train)
# 
#     # fit the model on the batches generated by datagen.flow()
#     history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
#                                   samples_per_epoch=x_train.shape[0], 
#                                   nb_epoch=nb_epoch, verbose=1,
#                                   validation_data=(x_test, y_test),
#                                   callbacks=[])
# 




