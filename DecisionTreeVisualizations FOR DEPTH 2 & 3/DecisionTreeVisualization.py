# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:55:46 2018

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn import tree
#graph visualization library
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image
import graphviz

dataset = pd.read_csv("datafile.csv")
print(dataset)

#coverting categorical data into numerical data becaise sk learn version of decision tree uses only numerical data
#and ny default it works on GINI INDEX

#converting binary columns
d = {'Yes': 1, 'No': 0}
dataset['Refund']=dataset['Refund'].map(d)
dataset['Cheat']=dataset['Cheat'].map(d)
#converting categorical columns
d = {'Single': 0, 'Married': 1, 'Divorced': 2}
dataset['Status']=dataset['Status'].map(d)
print(dataset)

features=list(dataset.columns[:3])
#converting features list into list of strings
[str(i) for i in features]


Y=dataset['Cheat']
X= dataset[features]

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(X,Y)

#converint label column into list of strings
Y=Y.astype("str")

#decision tree visualizations
dot_data = tree.export_graphviz(clf, out_file=None,  feature_names=features,class_names=Y,  filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data)  
graph.render('dtree_render',view=True)





#==============================================================================
#following is the generated diagraph of the decision tree 
#copy it and paste it on this link http://webgraphviz.com/ and click on generate it will generate the 
#decision tree for you


# digraph Tree {
# node [shape=box] ;
# 0 [label="X[2] <= 120.0\ngini = 0.497\nsamples = 13\nvalue = [6, 7]"] ;
# 1 [label="X[0] <= 0.5\ngini = 0.48\nsamples = 10\nvalue = [6, 4]"] ;
# 0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
# 2 [label="gini = 0.48\nsamples = 5\nvalue = [2, 3]"] ;
# 1 -> 2 ;
# 3 [label="gini = 0.32\nsamples = 5\nvalue = [4, 1]"] ;
# 1 -> 3 ;
# 4 [label="gini = 0.0\nsamples = 3\nvalue = [0, 3]"] ;
# 0 -> 4 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
# }
#==============================================================================

