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
                                     # (all pretrained on the ImageNet dataset)from PIL import Image




# folder
# /Users/user/Documents/workspace/MedicalML/project2/colorectal-histology-mnist

#----------------------------- Personal annotations a) ----------------------------------------
# how to read in tif images with python
# 
# print('reading in tif images')
#img_arr = plt.imread("colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/01_TUMOR/1A11_CRC-Prim-HE-07_022.tif_Row_601_Col_151.tif")
#view image
#plt.imshow(img_arr)
# plt.show()
#imarray = np.array(img_arr)
# check the size of the image as a np array 
# we have an image of 150x150 pixels
#print('shape of image array',imarray.shape)
# one can turn the array back into a PIL image like this:
#Image.fromarray(imarray)
#-----------------------------------------------------------------------------------------


#----------------------------- Personal annotations b) ----------------------------------------
# linux command for counting nb of lines in each directory
# for d in *; do ls $d | wc -l; done
# every category in the directory has 625 images
#-------------------------
# How to define a class and to call it
# obj = filename.classname
# obj = create_folders.(time_cos, cos,var[ind1],cutoff[0]) #shorttem = 40, numpolyterms = 4 filename.classname
#
#class classname:
# def __init__(self):
# def _function(self):
#-----------------------------------------------------------------------------------------

#------------------------------------------------ c) ----------------------------------------     
# 1. The initial path  ---- after downloading the image dataset:  
# colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000/01_TUMOR/,....,/08_EMPTY
# create train, test, valid directories starting with the existing dataset folder
# base_dir = 
# '/Users/user/Documents/workspace/MedicalML/project2/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000'
#
# 2. We need to create a work path    
# this creates us the folders: mainpath/train/                01_TUMOR/,...,08_EMPTY/
#                                      /validation/           01_TUMOR/,...,08_EMPTY/        
#                                      /test/                 01_TUMOR,/...08_EMPTY/ 
#-----------------------------------------------------------------------------------------                                           
 

class CreateFolders:


    def __init__(self,total_nb,nb_train,nb_test,nb_valid,categories):

        print('hello')

        self.base_dir = '/Users/user/Documents/workspace/MedicalML/project2/colorectal-histology-mnist/Kather_texture_2016_image_tiles_5000'

        print('creating working directories in folder',self.base_dir)
        #self.base_dir = '.'                    # should be there: in local directory

        self.categories = categories
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.validation_dir = os.path.join(self.base_dir, 'validation')
        self.test_dir = os.path.join(self.base_dir, 'test')
        self.total_nb = total_nb
        self.nb_train = nb_train
        self.nb_valid = nb_valid
        self.nb_test = nb_test

    def  directories(self):

        return self.train_dir,self.validation_dir,self.test_dir
		
    def init_path(self,ind_of_cat):
        # creates path to initial files for given category ind_of_cat

		# name of category(ind_of_cat)


        c = self.categories[ind_of_cat]
        path = self.base_dir + '/' + c
        return path

    def work_path(self,ind_of_cat,dir):
        # creates pathnames to work files. The work files will subsequently be copied to this path name
	    # dir is one of train, test or valid
        c = self.categories[ind_of_cat]
        path = self.base_dir + '/' + dir + '/' + c
        return path


    def get_folders(self):

        TRAIN = 'train'
        TEST = 'test'
        VALID = 'validation'

        work_folder = [TRAIN,TEST,VALID]
        work_ranges = [range(0,self.nb_train),range(self.nb_train,self.nb_train + self.nb_test),range(self.nb_train + self.nb_test,self.total_nb)]

        #-----------------------------------------------------------------------------------------
        # 1. create directory of the type base_dir/train,val,test
        #
        # into these directories we will then copy the files that are contained in each of the
        # subdirectories?
        #-----------------------------------------------------------------------------------------

        # create the directories if they do not already exist
        os.mkdir(self.train_dir)
        os.mkdir(self.validation_dir)
        os.mkdir(self.test_dir)


        #-----------------------------------------------------------------------------------------
        #
        # 2. create subdirectories for the different classes
        #
        # each of the 8 categories has to be split into train, test and dev sets
        #
        #    base_dir/train/cat1,....,cat8
        #    base_dir/test/cat1,....,cat8
        #    base_dir/valid/cat1,....,cat8
        #-----------------------------------------------------------------------------------------

        for dir in [self.train_dir, self.validation_dir, self.test_dir]:
           for name in self.categories:                 # classes
              subdir = os.path.join(dir,name)         # create directory of the type # path/train/
              os.mkdir(subdir)

        #-----------------------------------------------------------------------------------------
        # 3. copy files from initial to work folders
        #-----------------------------------------------------------------------------------------

        for i in range(len(self.categories)): # classes

            # list names of files contained in the initial path
            files_in_init = os.listdir(self.init_path(i))

            # copy a certain nb of files to train working directory
            for f in range(3):                        # each folder train,test,valid
                for j in work_ranges[f]:                # nb of images in train, test, valid folders
                   src = self.init_path(i) + '/' + files_in_init[j]
                   dst = self.work_path(i,work_folder[f]) + '/' + files_in_init[j] # path + filename
                   shutil.copyfile(src, dst)
	

        print('created working directory base_dir/train,test,valid/01_TUMOR/,...,08_EMPTY/')

        return self.train_dir,self.validation_dir,self.test_dir



