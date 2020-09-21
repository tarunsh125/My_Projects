# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:11:33 2020

@author: TA953608
"""

import datetime as dt
import pandas as pd
#from pandas import to_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

dataframe_train=pd.read_csv('train.csv')
dataframe_test=pd.read_csv('test.csv')
def classification_prep(dataframe_train):    
    '''
    dataframe_train=dataframe_train[[ 'Office_PIN','Applicant_City_PIN','Applicant_Occupation','Applicant_Qualification',
                                     'Manager_Joining_Designation','Manager_Current_Designation',
                                     'Manager_Grade','Manager_Status','Manager_Num_Application','Manager_Num_Coded',
                                     'Manager_Business','Manager_Num_Products','Manager_Business2','Manager_Num_Products2',
                                     'Business_Sourced']]# selecting proper dataframe
    '''
    print(dataframe_train['Manager_Joining_Designation'].dtypes)
    #print(dataframe_train['Manager_Joining_Designation'].select_dtypes(exclude=['int64']))
    #handling missing values and formatting
    dataframe_train.drop('ID', axis='columns', inplace=True)
    dataframe_train=dataframe_train.dropna(subset=['Manager_Grade'])
    #dataframe_train["Manager_DOJ"] = pd.to_numeric(dataframe_train["Manager_DOJ"])
    dataframe_train['Applicant_City_PIN']           =dataframe_train['Applicant_City_PIN'].fillna(dataframe_train['Applicant_City_PIN'].value_counts().index[0])
    dataframe_train['Applicant_Gender']             =dataframe_train['Applicant_Gender'].fillna(dataframe_train['Applicant_Gender'].value_counts().index[0])
    dataframe_train['Applicant_BirthDate']          =dataframe_train['Applicant_BirthDate'].fillna(dataframe_train['Applicant_BirthDate'].value_counts().index[0])
    dataframe_train['Applicant_Marital_Status']     =dataframe_train['Applicant_Marital_Status'].fillna(dataframe_train['Applicant_Marital_Status'].value_counts().index[0])
    dataframe_train['Applicant_Occupation']         =dataframe_train['Applicant_Occupation'].fillna('Sample')# as missing value is more, inserting another class
    dataframe_train['Applicant_Qualification']      =dataframe_train['Applicant_Qualification'].fillna(dataframe_train['Applicant_Qualification'].value_counts().index[0])
    dataframe_train['Manager_Joining_Designation']  =dataframe_train['Manager_Joining_Designation'].fillna(dataframe_train['Manager_Joining_Designation'].value_counts().index[0])
    dataframe_train['Manager_Current_Designation']  =dataframe_train['Manager_Current_Designation'].fillna(dataframe_train['Manager_Current_Designation'].value_counts().index[0])
    dataframe_train['Manager_Status']               =dataframe_train['Manager_Status'].fillna(dataframe_train['Manager_Status'].value_counts().index[0])
    dataframe_train['Manager_Joining_Designation']  =dataframe_train['Manager_Joining_Designation'].fillna(dataframe_train['Manager_Joining_Designation'].value_counts().index[0])
    Label_Encoder=LabelEncoder()       # label encoding the categorical variables
    dataframe_train['Applicant_Gender']=Label_Encoder.fit_transform(dataframe_train['Applicant_Gender'])
    dataframe_train['Applicant_Marital_Status']=Label_Encoder.fit_transform(dataframe_train['Applicant_Marital_Status'])
    dataframe_train['Applicant_Occupation']=Label_Encoder.fit_transform(dataframe_train['Applicant_Occupation'])
    dataframe_train['Applicant_Qualification']=Label_Encoder.fit_transform(dataframe_train['Applicant_Qualification'])
    dataframe_train['Manager_Joining_Designation']=Label_Encoder.fit_transform(dataframe_train['Manager_Joining_Designation'])
    dataframe_train['Manager_Current_Designation']=Label_Encoder.fit_transform(dataframe_train['Manager_Current_Designation'])
    dataframe_train['Manager_Grade']=Label_Encoder.fit_transform(dataframe_train['Manager_Grade'])
    dataframe_train['Manager_Status']=Label_Encoder.fit_transform(dataframe_train['Manager_Status'])
    dataframe_train['Manager_Gender']=Label_Encoder.fit_transform(dataframe_train['Manager_Gender'])
    
    
    dataframe_train["Application_Receipt_Date"] = pd.to_datetime(dataframe_train["Application_Receipt_Date"])
    dataframe_train["Application_Receipt_Date"] =dataframe_train["Application_Receipt_Date"].map(dt.datetime.toordinal)
    dataframe_train["Applicant_BirthDate"]      = pd.to_datetime(dataframe_train["Applicant_BirthDate"])
    dataframe_train["Applicant_BirthDate"]      =dataframe_train["Applicant_BirthDate"].map(dt.datetime.toordinal)
    dataframe_train["Manager_DOJ"]              = pd.to_datetime(dataframe_train["Manager_DOJ"])
    dataframe_train["Manager_DOJ"]              =dataframe_train["Manager_DOJ"].map(dt.datetime.toordinal)
    dataframe_train["Manager_DoB"]              = pd.to_datetime(dataframe_train["Manager_DoB"])
    dataframe_train["Manager_DoB"]              =dataframe_train["Manager_DoB"].map(dt.datetime.toordinal)
    print(dataframe_train.isnull().sum())
    #print(dataframe_train['Manager_DOJ'].head(5))
    #print(dataframe_train.isnull().sum())
    
    #fitting the model with Randomforest
    arr=dataframe_train.values
    x=arr[:,0:21]
    y=arr[:,21]
    return x,y

num_trees = 100 # defining the model
max_features = 3
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)


x_training,y_training=classification_prep(dataframe_train) #Preparing the training data
'''
x_train, x_test, y_train, y_test=train_test_split(x_training, y_training, test_size=0.30)#training the data with train test split

y_pred01,score01=classification_model(x_train, y_train)
print(score01.mean()) #initial training accuracy

y_pred02,score02=classification_model(x_test, y_test)
print(score02.mean())  #initial testing accuracy
'''
#score=roc_auc_score(y_test, y_pred)

model_fit=model.fit(x_training, y_training)
y_pred_training=model.predict(x_training)
score_training=cross_val_score(model,x_training, y_pred_training, scoring='roc_auc')
print(score_training.mean(),' ,  ', score_training.std())
roc_score=roc_auc_score(y_training, y_pred_training)
print('roc score =',roc_score.mean())
dataframe_test['Business_Sourced']=dataframe_test['Business_Sourced'].fillna(0)
x_output, y_output=classification_prep(dataframe_test)
y_pred=model.predict(x_output)
score_pred=cross_val_score(model,x_output, y_pred, scoring='roc_auc')
print(score_pred.mean(),' ,  ', score_pred.std())

ylist=list(y_pred)

for i in range(len(y_pred)) : dataframe_test.loc[i+1, 'Business_Sourced']=y_pred[i] 
dataframe_test.to_csv('test.csv')#, columns=['Business_Sourced'])
