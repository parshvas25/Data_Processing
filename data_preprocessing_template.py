#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:35:36 2019

@author: parshvashah
"""

#importing the libraries

import numpy
import matplotlib.pyplot
import pandas

# importing the dataset
dataset = pandas.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, -1].values

#take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean",axis =  0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3]) 

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#splitting our dataset into traning set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,
                                                    random_state = 0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)