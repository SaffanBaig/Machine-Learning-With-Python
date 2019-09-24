# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:33:17 2019

@author: Saffan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
# x contains all the independent variables
Y = dataset.iloc[:, 3].values
# y contains all the dependent variables

#missing data
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
#imputer = SimpleImputer(missing_values="NaN", strategy="mean")
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

#Categorical Data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Making training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)