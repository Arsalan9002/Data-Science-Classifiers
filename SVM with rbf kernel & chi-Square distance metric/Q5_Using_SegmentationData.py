# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 14:12:09 2018

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import chi2_kernel
from sklearn import preprocessing

dataset = pd.read_csv("segmentation.csv")


#class label encoding  done using map() function in numpy
d = {'BRICKFACE': 0, 'SKY': 1,'FOLIAGE':2,'CEMENT':3,'WINDOW':4,'PATH':5,'GRASS':6}
dataset['REGION-CENTROID-COL'] = dataset['REGION-CENTROID-COL'].map(d)
dataset[dataset < 0] = 0


#scaler = preprocessing.MinMaxScaler()
#scaled_df = scaler.fit_transform(dataset)
#scaled_df = pd.DataFrame(scaled_df)
#print(scaled_df)
#
#print(dataset.head(10))
array = dataset.values

#sepraratig the features and class labels
features = array[:,1:19]
features = features.astype('float')
target = array[:,0]
target = target.astype('float')

#data splitting
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(features.astype('int'), target.astype('int'), test_size=0.3, random_state=0)

#chi square distance metric
k = chi2_kernel(X_train,X_test,gamma = 0.5)

#pre computed kernel
model = svm.SVC(kernel='rbf', cache_size=1000, gamma= 0.5).fit(k, Y_train)

K = chi2_kernel(X_test)

pred = model.predict(K)

print("THE ACCURACY OF SEGMENTATION DATASET USING RANDOM 70/30 SPLIT IS {0} ".format(accuracy_score(Y_test,pred)))