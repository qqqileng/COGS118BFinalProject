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
import random
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn import tree
import graphviz
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

small_data = pd.read_csv('hmnist_8_8_L.csv')
small_colored_data = pd.read_csv('hmnist_8_8_RGB.csv')
medium_data = pd.read_csv('hmnist_28_28_L.csv')
medium_colored_data = pd.read_csv('hmnist_28_28_RGB.csv')
big_data = pd.read_csv('hmnist_64_64_L.csv')

sample_1 = medium_colored_data.drop("label", axis=1).values[0]
image_1 = sample_1.reshape((28, 28, 3))

sample = medium_colored_data.iloc[:,0:2352]
label = medium_colored_data['label']

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
#
#kmeans = KMeans(n_clusters=8).fit(medium_colored_data)
#kmeans_label = kmeans.labels_
#cm = confusion_matrix(label, kmeans_label)
#plt.imshow(cm,interpolation='none',cmap='Blues')
#for (i, j), z in np.ndenumerate(cm):
#    plt.text(j, i, z, ha='center', va='center')
#plt.xlabel("kmeans label")
#plt.ylabel("truth label")
#plt.show()
#
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(sample,label)
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data)

df_testing_1 = medium_colored_data.sample(frac = 0.2)
df_training_1 = pd.concat([medium_colored_data,df_testing_1],axis = 0).drop_duplicates(keep = False)
df_testing_2 = df_training_1.sample(frac = 0.25)
df_training_2 = pd.concat([medium_colored_data,df_testing_2],axis = 0).drop_duplicates(keep = False)
df_testing_3 = pd.concat([df_training_1,df_testing_2],axis = 0).drop_duplicates(keep = False).sample(frac = 0.3333)
df_training_3 = pd.concat([medium_colored_data,df_testing_3],axis = 0).drop_duplicates(keep = False)
df_testing_4 = pd.concat([df_training_1,df_testing_2,df_testing_3],axis = 0).drop_duplicates(keep = False).sample(frac = 0.5)
df_training_4 = pd.concat([medium_colored_data,df_testing_4],axis = 0).drop_duplicates(keep = False)
df_testing_5 = pd.concat([df_training_1,df_testing_2,df_testing_3,df_testing_4],axis = 0).drop_duplicates(keep = False)
df_training_5 = pd.concat([medium_colored_data,df_testing_5],axis = 0).drop_duplicates(keep = False)

training_sample_1 = df_training_1.iloc[:,0:2352]
training_label_1 = df_training_1['label']
testing_sample_1 = df_testing_1.iloc[:,0:2352]
testing_label_1 = df_testing_1['label']
training_sample_2 = df_training_2.iloc[:,0:2352]
training_label_2 = df_training_2['label']
testing_sample_2 = df_testing_2.iloc[:,0:2352]
testing_label_2 = df_testing_2['label']
training_sample_3 = df_training_3.iloc[:,0:2352]
training_label_3 = df_training_3['label']
testing_sample_3 = df_testing_3.iloc[:,0:2352]
testing_label_3 = df_testing_3['label']
training_sample_4 = df_training_4.iloc[:,0:2352]
training_label_4 = df_training_4['label']
testing_sample_4 = df_testing_4.iloc[:,0:2352]
testing_label_4 = df_testing_4['label']
training_sample_5 = df_training_5.iloc[:,0:2352]
training_label_5 = df_training_5['label']
testing_sample_5 = df_testing_5.iloc[:,0:2352]
testing_label_5 = df_testing_5['label']

clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0)
clf.fit(training_sample_1, training_label_1)
testing_predict_1 = pd.Series(clf.predict(testing_sample_1))
testing_compare_1 = pd.concat([testing_label_1.reset_index(drop=False),testing_predict_1],axis = 1)
clf.fit(training_sample_2, training_label_2)
testing_predict_2 = pd.Series(clf.predict(testing_sample_2))
testing_compare_2 = pd.concat([testing_label_2.reset_index(drop=False),testing_predict_2],axis = 1)
clf.fit(training_sample_3, training_label_3)
testing_predict_3 = pd.Series(clf.predict(testing_sample_3))
testing_compare_3 = pd.concat([testing_label_3.reset_index(drop=False),testing_predict_3],axis = 1)
clf.fit(training_sample_4, training_label_4)
testing_predict_4 = pd.Series(clf.predict(testing_sample_4))
testing_compare_4 = pd.concat([testing_label_4.reset_index(drop=False),testing_predict_4],axis = 1)
clf.fit(training_sample_5, training_label_5)
testing_predict_5 = pd.Series(clf.predict(testing_sample_5))
testing_compare_5 = pd.concat([testing_label_5.reset_index(drop=False),testing_predict_5],axis = 1)
testing_compare = pd.concat([testing_compare_1,testing_compare_2,testing_compare_3,testing_compare_4,testing_compare_5],axis = 0)

testing_compare_1a = testing_compare[(testing_compare['label'] == 1)].reset_index(drop=False)
testing_compare_1b = testing_compare[(testing_compare[0] == 1)].reset_index(drop=False)
testing_compare_2a = testing_compare[(testing_compare['label'] == 2)].reset_index(drop=False)
testing_compare_2b = testing_compare[(testing_compare[0] == 2)].reset_index(drop=False)
testing_compare_3a = testing_compare[(testing_compare['label'] == 3)].reset_index(drop=False)
testing_compare_3b = testing_compare[(testing_compare[0] == 3)].reset_index(drop=False)
testing_compare_4a = testing_compare[(testing_compare['label'] == 4)].reset_index(drop=False)
testing_compare_4b = testing_compare[(testing_compare[0] == 4)].reset_index(drop=False)
testing_compare_5a = testing_compare[(testing_compare['label'] == 5)].reset_index(drop=False)
testing_compare_5b = testing_compare[(testing_compare[0] == 5)].reset_index(drop=False)
testing_compare_6a = testing_compare[(testing_compare['label'] == 6)].reset_index(drop=False)
testing_compare_6b = testing_compare[(testing_compare[0] == 6)].reset_index(drop=False)
testing_compare_7a = testing_compare[(testing_compare['label'] == 7)].reset_index(drop=False)
testing_compare_7b = testing_compare[(testing_compare[0] == 7)].reset_index(drop=False)
testing_compare_8a = testing_compare[(testing_compare['label'] == 8)].reset_index(drop=False)
testing_compare_8b = testing_compare[(testing_compare[0] == 8)].reset_index(drop=False)
#pd.value_counts(testing_compare_1a[0].values, sort=False)