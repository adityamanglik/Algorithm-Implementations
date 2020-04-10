# -*- coding: utf-8 -*-
"""
Logistic Regression implementation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Logistic regression classifier. Trains using gradient descent
    
    Parameters:
        eta (float)       : Learning rate between 0.0 -> 1.0
        num_iter (int)    : Number of training iterations
        random_state (int): Random seed for weight initialization
        
        
    Attributes:
        weights_ (1-D vector): Parameters after fitting
        costs_ (list)        : Training cost storage for graph
    """

    def __init__(self, eta = 0.5, num_iter = 100, random_seed = 0):
        self.eta = eta
        self.num_iter = num_iter
        self.random_seed = random_seed
    
    def fit(self, X, y):
        """
        Train binary logistic regression model.
        
        Parameters
        ----------
        X: Input data of shape (number_of_samples x number_of_features)
        y: Output data of shape (number_of_samples x 1)

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_seed)
        self.weights_ = rgen.normal(loc = 0.0, scale = 0.1, size = 1 + X.shape[1])
        print('Shape of weights is ', self.weights_.shape)
        self.costs_ = []
        
        for iter in range(self.num_iter):
            z = self.net_input(X)
            output = self.activation(z)
            
            error = y - output
            #Update weights based on error
            self.weights_[1:] += self.eta * X.T.dot(error)
            self.weights_[0] += self.eta * sum(error)
            # Logistic cost function
            cost = sum(-y.dot(np.log(output)) - (1 - y).dot(1 - np.log(output)))
            self.costs_.append(cost)
    
    def net_input(self, X):
        """
        Calculates net input in the node

        Parameters
        ----------
        X : (1D vector)
            Input values.

        Returns
        -------
        Dot product of weights and input with bias term added.

        """
        return X.dot(self.weights_[1:]) + self.weights_[0]
    
    def activation(self, z):
        return 1 / (1 + np.exp(-1*z))
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1.0, 0.0)

#%%
# Test implementation
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
# Split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

#%%
X_subset = X_train[(y_train==0) | (y_train == 1)]
y_subset = y_train[(y_train==0) | (y_train == 1)]
classifier = LogisticRegression(eta = 0.05, num_iter = 1000, random_seed = 1)
classifier.fit(X_subset, y_subset)

plt.plot(classifier.costs_)
plt.show()