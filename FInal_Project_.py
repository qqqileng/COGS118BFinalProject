#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:29:10 2018

@author: QiLeng
"""

import pandas as pd
import numpy as np
#from neupy import algorithms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
#import tensorflow as tf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#mnist = input_data.read_data_sets('./hmnist_28_28_RGB.csv', one_hot = True)
medium_colored_data = pd.read_csv('./hmnist_64_64_L.csv')

medium_colored_data = medium_colored_data.values
np.random.shuffle(medium_colored_data)

scaler = StandardScaler()
rownum = len(medium_colored_data)
col = len(medium_colored_data[0])
X = medium_colored_data[:, 0:col-1]
Y = medium_colored_data[:, col-1]
pca = PCA(n_components=20)
pca.fit(X)
model = GaussianMixture(n_components=8,
                        covariance_type="full",
                        n_init=10,
                        random_state=0)

cluster = model.fit_predict(X)




    
    