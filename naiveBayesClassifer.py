# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math

types = {'Refund':str,'MaritalStatus':str,'TaxableIncome':int,'Evade':str}
names= ['Refund','MaritalStatus','TaxableIncome','Evade']
dataset = pd.read_csv("datafileAssign3.csv",usecols=names,dtype=types)
income_df=dataset['TaxableIncome']
print(dataset)
result = dataset['Refund'].value_counts()

def NormalDistribution(obj_varience,obj_mean,income):
    result = ((1/math.sqrt(2*math.pi*obj_varience))* math.exp(-(float(income)-float(obj_mean))**2/(2*obj_varience)))
    return result

def naiveBayes(refund,status,income):
    _list=[('refund',refund),('Status',status),('Income',income)]
    _list_yes_Class=[]
    _list_no_Class=[]
    for x,y in _list:
        if type(y)!= int and x == 'refund':
            #for evade class NO
            refund_prob1_arr = np.where((dataset['Refund'] == y) & (dataset['Evade'] == 'No'))
            
            refund_prob_1 = len(refund_prob1_arr[0])/result[0]
            _list_no_Class.append(refund_prob_1)
            #for evade class YES
            refund_prob2_arr = np.where((dataset['Refund'] == y) & (dataset['Evade'] == 'Yes'))
            refund_prob_2 = len(refund_prob2_arr[0])/result[1]
            _list_yes_Class.append(refund_prob_2)
            
        elif type(y)!= int and x == 'Status':
            #for evade class NO
            status_prob1_arr = np.where((dataset['MaritalStatus'] == y) & (dataset['Evade'] == 'No'))
            status_prob_1=len(status_prob1_arr[0])/result[0]
            _list_no_Class.append(status_prob_1)
            #for evade class YES
            status_prob2_arr = np.where((dataset['MaritalStatus'] == y) & (dataset['Evade'] == 'Yes'))
            status_prob_2=len(status_prob2_arr[0])/result[0]
            _list_yes_Class.append(status_prob_2)
            
        elif  type(y) == int and x == 'Income':
            #income-evade yes varience and mean
            evade_yes_count = np.where((dataset['Evade'] == 'No'))
            varience_yes= np.var(income_df[evade_yes_count[0].tolist()],ddof=1)
            mean_yes = np.mean(income_df[evade_yes_count[0].tolist()])
            Prob_no_dsitribution = NormalDistribution(varience_yes,mean_yes,y)
            _list_no_Class.append(Prob_no_dsitribution)
            #income-evade No varience & mean
            evade_no_count = np.where((dataset['Evade'] == 'Yes'))
            varience_no= np.var(income_df[evade_no_count[0].tolist()],ddof=1)
            mean_no = np.mean(income_df[evade_no_count[0].tolist()])
            Prob_yes_dsitribution = NormalDistribution(varience_no,mean_no,y)
            _list_yes_Class.append(Prob_yes_dsitribution)
            
    class_no_final = np.prod(_list_no_Class)
    class_yes_final = np.prod(_list_yes_Class)
    if class_no_final > class_yes_final:
        print("Given Test example belongs to CLASS: NO")
    else:
        print("Given Test example belongs to CLASS: YES")
            
naiveBayes('Yes','Single',110)
naiveBayes('Yes','Single',5)
naiveBayes('Yes','Married',95)
naiveBayes('No','Divorced',63)