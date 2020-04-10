import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self._DataShuffled = False
        self.cost_track = []

    def initialize_weights(self, m):
        # create random weights
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.01, scale=0.01, size = m + 1)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def fit(self, X, y):
        self.initialize_weights(X.shape[1])

        #run fit over data for n_iter
        for i in range(self.n_iter):
            errors = 0
            output = self.activation(self.net_input(X))
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0]  += self.eta * errors.sum()
            cost = np.dot(errors, errors.T)
            self.cost_track.append(cost)
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(y.size)
        return X[r], y[r]

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0]  += self.eta * error
        cost = 0.5 * error**2
        return cost

    def Stochastic_fit(self, X, y):
        self.initialize_weights(X.shape[1])

        for i in range(self.n_iter):
            if self._DataShuffled !=True:
                X, y = self._shuffle(X, y)
                self._DataShuffled = True
            costs = []
            for xi, yi in zip(X, y):
                costs.append(self._update_weights(xi, yi))
            self.cost_track.append(sum(costs)/len(costs))
        return self


dataset = pd.read_csv('iris.data', header=None)
print(dataset.tail())

y = dataset.iloc[:100, 4].values
y = np.where(y == 'Iris-setosa', 1, -1)

X = dataset.iloc[:100, [0, 2]].values

fig, ax = plt.subplots(nrows= 1 , ncols= 3, figsize=(15,4))
ada1 = AdalineGD(eta=0.01, n_iter = 10, random_state= 1).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_track) +1), np.log10(ada1.cost_track), marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error')
ax[0].set_title('Adaline, learning rate = 0.01')

ada2 = AdalineGD(eta = 0.0001, n_iter = 10, random_state= 1).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_track) +1), np.log10(ada2.cost_track), marker = 'x')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error')
ax[1].set_title('Adaline, learning rate = 0.1')

# Standardization
X_s = np.copy(X)
X_mean, X_std = np.mean(X_s, axis=0), np.std(X_s, axis = 0)
X_s -= X_mean
X_s /= X_std
print(X, X_s)

ada3 = AdalineGD(eta = 0.01, n_iter = 10, random_state= 1).fit(X_s,y)
ax[2].plot(range(1, len(ada3.cost_track) +1), np.log10(ada3.cost_track), marker = 'o')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('log(Sum-squared-error')
ax[2].set_title('Adaline, learning rate = 0.1')
plt.show()

ada4 = AdalineGD(eta = 0.1, n_iter = 10, random_state= 1).Stochastic_fit(X_s,y)
fig, ax = plt.subplots()
ax.plot(range(1, len(ada4.cost_track) +1), np.log10(ada4.cost_track), marker = 'x')
ax.set_xlabel('Epochs')
ax.set_ylabel('log(Sum-squared-error')
ax.set_title('Adaline, learning rate = 0.1')
plt.show()