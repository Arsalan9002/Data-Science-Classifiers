# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:09:36 2018

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import chi2_kernel


dataset = pd.read_csv("dermatology.csv")
array = dataset.values

features = array[:,1:19]
features = features.astype('float')
target = array[:,0]
#print(target)
target = target.astype('float')

#data splitting
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(features.astype('int'), target.astype('int'), test_size=0.3, random_state=0)
#clf = svm.SVC()
#clf.fit(features, target)

k = chi2_kernel(X_train,X_test,gamma = 0.5)
model = svm.SVC(kernel='linear', cache_size=1000, gamma= 0.5).fit(k, Y_train)
K = chi2_kernel(X_test)
pred = model.predict(K)
print("THE ACCURACY OF DERMATOLOGY DATASET USING RANDOM 70/30 SPLIT IS {0} ".format(accuracy_score(Y_test,pred)))