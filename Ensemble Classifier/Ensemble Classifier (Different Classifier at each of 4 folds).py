# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn import tree
from sklearn.svm import SVC
from sklearn.cluster import KMeans



######################################Questio1######################################################
df = pd.DataFrame({'Name':[' ','Ali','Ahmed','Nida',' '],'Field':['C','E','E','C','C'],'Age':['','','','',''],
                   'Marks':[-90,60,-10,70,75]})
print(df)


#i
del df['Age']
print(df)

#ii
df['Name']=df['Name'].replace(' ','---')
print(df)

#iii
df['Field'] = df['Field'].replace('C',0)
df['Field'] = df['Field'].replace('C',1)
#print(df)
#print(df.info())

#iv
#df['Marks']= df['Marks'].apply(lambda x : x.replace(np.mean()))
#print(df)







###############################Question2#################################################
digits= datasets.load_digits()

features= digits.data
target= digits.target
print("length of samples is ",len(features))
print("target is",target)


x=features
y=target

#==============================================================================
# dt= tree.DecisionTreeClassifier()
# _dt_list =cross_validation.cross_val_score(dt,x,y,cv=4)
# print(_dt_list)
# 
# clf = SVC(C=1,kernel='linear')
# _svm_list =cross_validation.cross_val_score(clf,x,y,cv=4)
# print(_svm_list)
# 
# nb = GaussianNB()
# _nb_list =cross_validation.cross_val_score(nb,x,y,cv=4)
# print(_nb_list)
#==============================================================================
#calling models
dt= tree.DecisionTreeClassifier()
clf = SVC(C=1,kernel='linear')
nb = GaussianNB()
km=KMeans()
knn=KNeighborsClassifier(n_neighbors=3)
i=1
kf = KFold(len(x),n_folds=4)

for train_index,test_index in kf:
    X_train,X_test =x[train_index],x[test_index]
    Y_train,Y_test=y[train_index],y[test_index]
    if i==1:
        
        model1=dt.fit(X_train,Y_train)
        prediction_dt=dt.predict(X_test)
        print("accuracy using decision tree ",accuracy_score(Y_test,prediction_dt))
        i+=1
    elif i==2:
       
        model2=clf.fit(X_train,Y_train)
        prediction_svm=clf.predict(X_test)
        print("accuracy using svm ",accuracy_score(Y_test,prediction_svm))
        i+=1
        
    elif i==3:
        
        model3=nb.fit(X_train,Y_train)
        prediction_nb=nb.predict(X_test)
        print("accuracy using naive bayes ",accuracy_score(Y_test,prediction_nb))
        i+=1
    
    elif i==4:
        model_dum =km.fit(X_train,Y_train)
        prediction_km=km.predict(X_test)
        print("accuracy using kmeans ",accuracy_score(Y_test,prediction_km))
        
        model4=knn.fit(X_train,Y_train)
        prediction_knn=knn.predict(X_test)
        print("accuracy using knn ",accuracy_score(Y_test,prediction_knn))
        i+=1
        
        




