# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:16:27 2018

@author: QiLeng 
"""

import pandas as pd
import numpy as np
from neupy import algorithms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
#import arrow
#from tqdm import tqdm
#import re
#from IPython.display import Image
#import datetime
#import seaborn as sns
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from random import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# We are using the dataset 'hmnist_28_28_RGB.csv' which contains the RGB 
# data of each image with 28 pixel * 28 pixel.
medium_colored_data = pd.read_csv('./hmnist_28_28_RGB.csv')

# Explore the head of file
#medium_colored_data.head()
#medium_colored_data.shape
#sample_1 = medium_colored_data.drop("label", axis=1).values[0]
#image_1 = sample_1.reshape((28, 28, 3)) # Reshape each row to be in 3 channels

# Display the first image 
#fig, ax = plt.subplots(1,4,figsize=(20,5))
#for channel in range(3):
#    ax[channel].imshow(image_1[:,:,channel], cmap="gray")
#    ax[channel].set_title("Channel {}".format(channel+1))
#    ax[channel].set_xlabel("Width")
#    ax[channel].set_ylabel("Height")
#ax[3].imshow(image_1)
#ax[3].set_title("All channels together")
#ax[3].set_xlabel("Width")
#ax[3].set_ylabel("Height")

#----------------------- LVQ Algorithm --------------------------

# Function Description: In this algorithm, we will use Euclidean distance 
# to measure the affinity between two data points.
# Define the function for calculation of Euclidean distance
#def Euclidean_Distance(data1, data2):
#    distance = 0.0
#    for i in range(len(data1[0])-1): # The last value is the label
#        distance += ( data1[i] - data2[i] )**2
#    return sqrt(distance)
#
## Function Description: This function calculates the distance between the 
## codebook and other data points.
#def Best_Codebook(codebooks, row):
#    distances = []
#    for i in range(len(codebooks)):
#        distances.append((codebooks[i], Euclidean_Distance(codebooks[i], row)))
#    # Sort the distance in decreasing order
#    distances.sort(key=lambda tup: tup[1])
#    return distances[0][0]




 
# Shuffle the dataset
medium_colored_data = medium_colored_data.values
np.random.shuffle(medium_colored_data)

# Split the X and Y values. Do a 5-folds cross validation.
rownum = len(medium_colored_data)
col = len(medium_colored_data[0])
X = medium_colored_data[:, 0:col-1]
Y = medium_colored_data[:, col-1]
kf = KFold(n_splits=5)
kf.get_n_splits(X)
print(kf)

# Use a for loop to perform the LVQ 5 times for different sets of training and 
# testing data.
count = 0
acc = [None] * 5
Y = Y-1
'''
X = X[0:100,0:30]
Y = Y[0:100]
transformer = Normalizer().fit(X)
X = transformer.transform(X)
lvqnet = algorithms.LVQ2(n_inputs=30,n_classes=8)
lvqnet.train(X,Y, epochs=100)
Y_pred = lvqnet.predict(X)
acc = accuracy_score(Y,Y_pred)
'''

for train_index, test_index in kf.split(X):
    # Set up the training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    print(X_train.shape)
    print(Y_train.shape)
    
    # Train the algorithm using training set
    lvqnet = algorithms.LVQ3(n_inputs=col-1, n_classes=8)
    lvqnet.train(X_train, Y_train, epochs=500)
    
    # Test the algororithm using testing set
    Y_pred = lvqnet.predict(X_test)
    
    acc[count] = accuracy_score(Y_test, Y_pred)
    count = count + 1

    

    # Initialize the codebook vectors to be vectors that are randomly selected 
    # from the training data we have. 
#    codebooks = []
#    randindex = np.random.randint(0, len(X[0])-1)
#    codebooks.append(X[randindex,:])
#    for x in range (1,8):
#        randindex = np.random.randint(0, len(X[0])-1)
#        codebooks.append(X[randindex,:].reshape(-1))
#        
    # After the initialization, we need to train the codebook vector to best 
    # fit the dataset
    
    
    # Find the best matching codebook for each new data point
    
    
# Next, we need to find the best matching unit (codebook vector)
    