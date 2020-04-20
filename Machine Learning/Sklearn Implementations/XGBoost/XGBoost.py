#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:01:59 2020

@author: admangli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#%%
# Convert data labels to features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_enc_country = LabelEncoder()
X[:, 1] = label_enc_country.fit_transform(X[:, 1])
label_enc_gender = LabelEncoder()
X[:, 2] = label_enc_gender.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1, 2])], remainder='passthrough')
X = columnTransformer.fit_transform(X)
# Avoid dummy variable trap
X = X[:, 1:]
X = X[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]]
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

#%%
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# print(classifier.score(y_test.reshape(-1, 1), y_pred.reshape(-1, 1)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#%%
# K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X_train, y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())

#%%

# RFC for comparison
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# print(classifier.score(y_test.reshape(-1, 1), y_pred.reshape(-1, 1)))

from sklearn.metrics import confusion_matrix
cm_rfc = confusion_matrix(y_test, y_pred)

accuracies_rfc = cross_val_score(classifier, X_train, y_train, cv = 10)
print(accuracies_rfc.mean())
print(accuracies_rfc.std())

#%%
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.01,
                                        verbose = 1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# print(classifier.score(y_test.reshape(-1, 1), y_pred.reshape(-1, 1)))

from sklearn.metrics import confusion_matrix
cm_gbm = confusion_matrix(y_test, y_pred)

accuracies_gbm = cross_val_score(classifier, X_train, y_train, cv = 10)
print(accuracies_gbm.mean())
print(accuracies_gbm.std())