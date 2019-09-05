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
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Categorical Data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y[:, 0] = labelencoder_Y.fit_transform(Y[:, 0])