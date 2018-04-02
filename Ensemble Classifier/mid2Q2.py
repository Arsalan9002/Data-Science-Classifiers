# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:57:34 2018

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC

dataset = pd.read_csv("german_train.csv")
array = dataset.values


features = array[:,0:24]
features.astype('int')
target=array[:,24]
target.astype('int')

print(features)
print(target)


#X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(features.astype('int'), target.astype('int'), test_size=0.3, random_state=0)


nb = GaussianNB()
model1 =nb.fit(features,target)
predictions = nb.predict(features)
print(accuracy_score(target,predictions))


X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(features,target,test_size=0.33,random_state=1)
dt = tree.DecisionTreeClassifier()
model2 = dt.fit(X_train,Y_train)
predictions = dt.predict(X_test)
print(accuracy_score(Y_test,predictions))

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(features,target,test_size=0.33,random_state=2)
dt = tree.DecisionTreeClassifier()
model3 = dt.fit(X_train,Y_train)
predictions = dt.predict(X_test)
print(accuracy_score(Y_test,predictions))


svm = SVC()
model4= svm.fit(features,target)
predictions = svm.predict(features)
print(accuracy_score(target,predictions))


dataset = pd.read_csv("german_test.csv")
array = dataset.values
features_test =array[:,0:24]
features.astype('int')
target_test = array[:,24]

#model1.fit(features,target)
prediction1= model1.predict(features_test)
print(prediction1)
print(accuracy_score(target_test,prediction1))

prediction2= model2.predict(features_test)
print(prediction2)
print(accuracy_score(target_test,prediction2))
#knn = KNeighborsClassifier()

prediction3= model3.predict(features_test)
print(prediction3)
print(accuracy_score(target_test,prediction3))

prediction4= model4.predict(features_test)
print(prediction4)
print(accuracy_score(target_test,prediction4))

_MeanAccuracy=[]
_temp =[]
length = len(prediction1)
i=0

while i<length:
    _temp.append(prediction1[i])
    _temp.append(prediction2[i])
    _temp.append(prediction3[i])
    _temp.append(prediction4[i])
    
    count1= _temp.count(1)
    count2= _temp.count(2)
    if count1 > count2:
        _MeanAccuracy.append(1)
        i+=1
        _temp=[]
    else:
        _MeanAccuracy.append(1)
        i+=1
        _temp=[]

print(accuracy_score(target_test,_MeanAccuracy))

        
    
    
#print(dataset)


