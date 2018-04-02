# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:33:27 2018

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB



data = {'cities' : ['lahore','karachi'], 'provinces' : ['punjab','sindh']}
frame1 = pd.DataFrame(data)
#print(frame1)

frame2= pd.read_json("data.json")
#print(frame2)

frame1=frame1.append(frame2, ignore_index=True)
frame1=frame1.drop_duplicates();
#without case
#frame1= frame1.iloc[frame1.provinces.str.lower().argsort()]
#with case
frame1 = frame1.sort_values(by='provinces')
frame3= frame1.reset_index(drop=True)

print(frame3)