import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import datatset
dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode categorical variables and OneHotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_x = LabelEncoder()
X[:, 3] = labelEncoder_x.fit_transform(X[:, 3])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Avoiding dummy variable trap
X = X[:, 1:] # We do this to remove columns not contributing information

# Split dataset into train-test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Fit model to training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Get prediction vector
y_pred = regressor.predict(X_test)
print(y_pred, y_test)
print(regressor.score(X_test, y_test))
#%%
# Append ones column to X for bias term in regression
X_opt = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1).astype(float)
X_opt = X_opt[:, [0, 1,2, 3, 4, 5]]
#%%
# Backward elimination to build optimal regression model
import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
#%%
X_opt = X_opt[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
#%%
X_opt = X_opt[:, [0, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
#%%
X_opt = X_opt[:, [0, 1, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
#%%
X_opt = X_opt[:, [0, 1]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

#%% Forward selection linear regression
