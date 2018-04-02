# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:57:34 2018

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv("german_test.csv")
array = dataset.values

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(features.astype('int'), target.astype('int'), test_size=0.3, random_state=0)

print(dataset)