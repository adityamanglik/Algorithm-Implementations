#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:51:33 2020

@author: admangli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#%%
# from sklearn.model_selection import train_test_split
# As dataset is very small, we need entire dataset for a good fit. Further, it is simply a model of data.

# Build a linear regression model
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

#%%
#Visualizing linear regressor
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title('Truth or bluff (linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#%%
# Build a polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

polynomialRegressor = LinearRegression()
polynomialRegressor.fit(X_poly, y)

#%%
#Visualizing polynomial regressor
# plt.plot(X, polynomialRegressor.predict(X), color='green')
X_grid = np.arange(min(X), max(X), 0.1).reshape((90, 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, polynomialRegressor.predict(poly_reg.fit_transform(X_grid)), color='yellow')
plt.title('Truth or bluff (linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#%%

pred = np.array([6.5]).reshape((1, -1))
print(linear_regressor.predict(pred))

print(polynomialRegressor.predict(poly_reg.fit_transform(pred)))


