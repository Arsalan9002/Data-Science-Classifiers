# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:52:37 2018

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn import tree


dataset = pd.read_csv("segmentation.csv")
d={'BRICKFACE':1,'SKY':2,'FOLIAGE':3,'CEMENT':4,'WINDOW':5,'PATH':6,'GRASS':7}
dataset['REGION-CENTROID-COL'] = dataset['REGION-CENTROID-COL'].map(d)
array = dataset.values
#print(df['REGION-CENTROID-COL'])
#***********************************************************************#
#random train/test split(70%, 30%) with dermatology data
features = array[:,1:19]
features = features.astype('float')
target = array[:,0]
#print(target)
target = target.astype('float')

#data splitting
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(features.astype('int'), target.astype('int'), test_size=0.3, random_state=0)
#Naive bayes model with  random train/test split (70/30%) using dermatology data (accuracy and f-measure)
nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_test)
nb_random_accuracy=accuracy_score(Y_test, predictions)
print("Accuracy using Naive bayes with random train/test (70/30) using dermatology data is {0} ".format(nb_random_accuracy))
print("F-measure using Naive Bayes with random train/test (70/30) using dermatology data is {0} ".format(classification_report(Y_test, predictions)))
print("\n")

#KNN with  random train/test split (70/30%) using dermatology data (accuracy and f-measure)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
knn_random_accuracy=accuracy_score(Y_test, predictions)
print("Accuracy using knn with random train/test (70/30) using dermatology data is {0} ".format(knn_random_accuracy))
print("F-measure using knn with random train/test (70/30) using dermatology data is {0} ".format(classification_report(Y_test, predictions)))
print("\n")

#C 4.5 with  random train/test split (70/30%) using dermatology data (accuracy and f-measure)
clf = tree.DecisionTreeClassifier(random_state=0, criterion = 'entropy',max_depth=3)
clf = clf.fit(X_train,Y_train)
predictions = clf.predict(X_test)
decision_random_accuracy = accuracy_score(Y_test, predictions)
print("Accuracy using C4.5 with random train/test (70/30) using dermatology data is {0} ".format(decision_random_accuracy))
print("F-measure using C4.5 with random train/test (70/30) using dermatology data is {0} ".format(classification_report(Y_test, predictions)))
print("\n")

#==============================================================================
#10 fold in loop using Naive Bayes

kf= KFold(n_splits=10)
_MeanList_nb=[]
i=1
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]
    nb.fit(X_train,y_train)
    y_predict_test = nb.predict(X_test)
    _MeanList_nb.append(accuracy_score(y_test, y_predict_test))
    i+=1
    
print("Accuracy using Naive Bayes for 10 fold using dermatology data is {0} ".format(np.mean(_MeanList_nb)))
print("F-measure using Naive Bayes for 10 fold  fold using dermatology data is {0} ".format(f1_score(y_test, y_predict_test,average=None)))
print("\n")
#10 fold in loop using knn
kf = KFold(n_splits=10)
_MeanList_knn=[]
i=1
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    #print(X_train)
    y_train, y_test = target[train_index], target[test_index]
    knn.fit(X_train,y_train)
    y_predict_test = knn.predict(X_test)
    _MeanList_knn.append(accuracy_score(y_test, y_predict_test))
    i+=1
    
print("Accuracy using knn for 10 fold using dermatology data is {0} ".format(_MeanList_knn))
print("F-measure using knn for 10 fold using dermatology data is {0} ".format(f1_score(y_test, y_predict_test,average=None)))
print("\n") 

#10 fold in loop using decision tree
kf = KFold(n_splits=10)
i=1
_MeanList_decision=[]
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = target[train_index], target[test_index]
    clf.fit(X_train,y_train)
    y_predict_test = clf.predict(X_test)
    _MeanList_decision.append(accuracy_score(y_test, y_predict_test))
    i+=1
    
print("Mean Accuracy using knn for 10 fold using dermatology data is {0} ".format(np.mean(_MeanList_decision)))
print("F-measure using knn for 10 fold using dermatology data is {0} ".format(f1_score(y_test, y_predict_test,average=None)))
print("\n")

accuPlot = pd.DataFrame({'accuracy':[nb_random_accuracy,knn_random_accuracy,decision_random_accuracy,np.mean(_MeanList_nb),np.mean(_MeanList_knn),np.mean(_MeanList_decision)]})
accuPlot.index = ['accuracy_nb_70/30','accuracy_knn_70/30','accuracy_C4.5_70/30','accuracy_nb_10fold','accuracy_knn_10fold','accuracy_C4.5_10fold']
accuPlot.plot(kind = 'barh',title = "Accuracy Plot")