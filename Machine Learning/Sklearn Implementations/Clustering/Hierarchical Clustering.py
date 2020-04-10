#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:50:01 2020

@author: admangli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:].values
#%%
#Using the dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Distances')
plt.show()

#%% Fitting hierarchical clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
clusterer = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = clusterer.fit_predict(X)
#%% Visualize results
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, color = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, color = 'yellow', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, color = 'blue', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, color = 'green', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, color = 'black', label = 'Cluster 5')
# plt.scatter(clusterer.get_params, X[y_hc == 0, 1], s = 200, color = 'magenta', label = 'Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Agglomerative clusters')
plt.legend()