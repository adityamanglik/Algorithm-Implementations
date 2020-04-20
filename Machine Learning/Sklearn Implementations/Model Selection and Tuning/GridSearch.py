#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:33:02 2020

@author: admangli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

#%%
# Scale values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

#%%

from sklearn.svm import SVC
classifier = SVC(C = 1, kernel = 'rbf', gamma = 0.8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classifier.score(X_test, y_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#%%
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X_train, y_train, cv = 10, n_jobs = -1, verbose = 1)
print(accuracies.mean())
print(accuracies.std())

#%%
# Apply grid search to find the right model for job
from sklearn.model_selection import GridSearchCV
parameters = [
    {'C' : [1, 10, 100, 1000], 'kernel' : ['linear'],}, # This tells if the problem is linear or non-linear
    {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma':np.linspace(0.1, 1.0, 10)}
             ]
grid_searcher = GridSearchCV(estimator = classifier, param_grid = parameters, 
                             scoring = 'accuracy', n_jobs = -1, cv = 10, verbose = 1)

grid_searcher = grid_searcher.fit(X_train, y_train)
print(grid_searcher.best_score_)
best_parameters = grid_searcher.best_params_

#%% Training plot
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = .01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = .01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'cyan')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('darkred', 'darkgreen', 'blue'))(i), label = j)
plt.title('Training set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

#%% Test plot
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = .01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = .01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('darkred', 'darkgreen'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()