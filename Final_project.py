# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:16:27 2018

@author: zhw18
"""

import pandas as pd
import numpy as np
import arrow
from tqdm import tqdm
import re
from IPython.display import Image
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

small_data = pd.read_csv('D:/Machine Learning/Colorectal Histology/hmnist_8_8_L.csv')
small_colored_data = pd.read_csv('D:/Machine Learning/Colorectal Histology/hmnist_8_8_RGB.csv')
medium_data = pd.read_csv('D:/Machine Learning/Colorectal Histology/hmnist_28_28_L.csv')
medium_colored_data = pd.read_csv('D:/Machine Learning/Colorectal Histology/hmnist_28_28_RGB.csv')
big_data = pd.read_csv('D:/Machine Learning/Colorectal Histology/hmnist_64_64_L.csv')

medium_colored_data.head()
medium_colored_data.shape
sample_1 = medium_colored_data.drop("label", axis=1).values[0]
image_1 = sample_1.reshape((28, 28, 3))

fig, ax = plt.subplots(1,4,figsize=(20,5))
for channel in range(3):
    ax[channel].imshow(image_1[:,:,channel], cmap="gray")
    ax[channel].set_title("Channel {}".format(channel+1))
    ax[channel].set_xlabel("Width")
    ax[channel].set_ylabel("Height")
ax[3].imshow(image_1)
ax[3].set_title("All channels together")
ax[3].set_xlabel("Width")
ax[3].set_ylabel("Height")