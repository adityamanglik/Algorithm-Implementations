#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:21:22 2020

@author: admangli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values
#%%
#feature scaling is compulsory for SVR
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)
#%%
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)
print(y_scaler.inverse_transform(regressor.predict(X_scaler.transform(np.array([[6.5]])))))
#%%
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('SVR')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
#%%
X_grid = X_scaler.transform(np.arange(1, 11, 0.01).reshape(-1, 1))
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()