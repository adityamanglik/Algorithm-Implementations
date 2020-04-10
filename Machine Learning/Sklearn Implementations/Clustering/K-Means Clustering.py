#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:55:18 2020

@author: admangli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
print(dataset.head())

X = dataset.iloc[:, 3:].values
#%%
# Convert categorical variables
from sklearn.cluster import KMeans
WCSS = []
for i in range(1, 11):
    clusterer = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 100, verbose = 1)
    clusterer.fit(X)
    WCSS.append(clusterer.inertia_)
    
#%%
plt.plot(range(1, 11), WCSS)
plt.title('WCSS Elbow curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS / Inertia')
plt.show()

#%%
# Apply K means to whole dataset to see the clusters
clusterer = KMeans(n_clusters = 5)
y_kmeans = clusterer.fit_predict(X)

#%% Visualizing the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, color = 'red', label = 'High Rollers')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, color = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, color = 'green', label = 'Careless')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, color = 'yellow', label = 'Careful')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, color = 'black', label = 'Sensible')
plt.scatter(clusterer.cluster_centers_[:, 0], clusterer.cluster_centers_[:, 1], s = 200, color = 'magenta', label = 'Centroids')
plt.title('Clusters generated')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()