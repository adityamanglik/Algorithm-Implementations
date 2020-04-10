#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:43:29 2020

@author: admangli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
#%%
# Import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#%%
# Encode categorical features to numerical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_country = LabelEncoder()
labelencoder_gender = LabelEncoder()
X[:, 1] = labelencoder_country.fit_transform(X[:, 1])
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])
# Avoid dummy variable trap
X = X[:, 1:]
#%%
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)

X = ct.fit_transform(X)

#%%
#Scale dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

#%%
from keras.models import Sequential
from keras.layers import Dense

nn = Sequential() #When we add, we add the first layer after inputs, i.e., the first hidden layer
nn.add(Dense(units = 6, kernel_initializer = 'uniform', 
             activation = "relu", input_shape = (10,)))
# Add one more layer
nn.add(Dense(units = 6, activation = "relu", kernel_initializer = 'uniform'))
# Add output layer
nn.add(Dense(units = 1, kernel_initializer = "uniform", activation = 'sigmoid'))

nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
nn.fit(X_train, y_train, epochs = 100, batch_size = 10, use_multiprocessing = True)

#%%
# Predict on test values
predictions = nn.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix
binary_predictions = (predictions > 0.5)
cm = confusion_matrix(y_test, binary_predictions)
#%%
# Compare NN to Kernel SVM and RFC
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200)
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))