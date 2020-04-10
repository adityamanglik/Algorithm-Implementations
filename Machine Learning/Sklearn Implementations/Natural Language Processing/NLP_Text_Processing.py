#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:24:06 2020

@author: admangli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

text = pd.read_csv('Restaurant_Reviews.tsv', sep = '\t', quoting = 3)

#%%
# Cleaning text
import re

import nltk
stop_words = nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()

#%%
cleaned_text = []

for sentence in text['Review']:
    stopless_sentence = re.sub(r'[^a-zA-Z]', ' ', sentence) # Eliminates punctuation
    lowerCase_sentence = stopless_sentence.lower() # Convert all characters to lowercase
    list_of_words = lowerCase_sentence.split() # Split by whitespace
    # eliminate redundant words and stop words, stem each word
    cleaned_words = [porter_stemmer.stem(word) for word in list_of_words if not word in set(stopwords.words('english'))]
    # Convert the list of words back into cleaned text string
    cleaned_sentence = ' '.join(cleaned_words)
    cleaned_text.append(cleaned_sentence)
    
#%%
# Create a bag of words model via tokenization
from sklearn.feature_extraction.text import CountVectorizer
X = CountVectorizer(max_features = 1500).fit_transform(cleaned_text).toarray()
y = text.iloc[:, 1].values

#%%

#Split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#%%

#Compare multiple classification models from checking performance
# Logistic Regression, Naive Bayes, SVM, Kernel SVM, RandomForest
# Determine score based on Confusion Matrix, not score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
cm = confusion_matrix(y_test, lr_classifier.predict(X_test))
print('Logistic Regression', lr_classifier.score(X_test, y_test), (cm[0, 1] + cm[1, 0])/y_test.shape[0])
precision = cm[1, 1]/(cm[1, 1]+cm[0, 1])
recall = cm[1, 1]/(cm[1, 1]+cm[1, 0])
print(precision, recall, 'n', 'F1', 2*precision*recall/(precision+recall))
#%%

from sklearn.svm import SVC
svc_classifier = SVC(kernel = 'linear')
svc_classifier.fit(X_train, y_train)
cm = confusion_matrix(y_test, svc_classifier.predict(X_test))
print('Linear SVM', svc_classifier.score(X_test, y_test), cm[0, 1] + cm[1, 0])
precision = cm[1, 1]/(cm[1, 1]+cm[0, 1])
recall = cm[1, 1]/(cm[1, 1]+cm[1, 0])
print(precision, recall, 'n', 'F1', 2*precision*recall/(precision+recall))
#%%

from sklearn.svm import SVC
svck_classifier = SVC()
svck_classifier.fit(X_train, y_train)
cm = confusion_matrix(y_test, svck_classifier.predict(X_test))
print('Kernel SVM', svck_classifier.score(X_test, y_test), cm[0, 1] + cm[1, 0])
precision = cm[1, 1]/(cm[1, 1]+cm[0, 1])
recall = cm[1, 1]/(cm[1, 1]+cm[1, 0])
print(precision, recall, 'n', 'F1', 2*precision*recall/(precision+recall))
#%%
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
cm = confusion_matrix(y_test, nb_classifier.predict(X_test))
print('Naive Bayes', nb_classifier.score(X_test, y_test), cm[0, 1] + cm[1, 0])
precision = cm[1, 1]/(cm[1, 1]+cm[0, 1])
recall = cm[1, 1]/(cm[1, 1]+cm[1, 0])
print(precision, recall, 'n', 'F1', 2*precision*recall/(precision+recall))
#%%
from sklearn.ensemble import RandomForestClassifier
for i in range(100, 1000, 100):
    rfc_classifier = RandomForestClassifier(n_estimators = i, max_depth = 2)
    rfc_classifier.fit(X_train, y_train)
    cm = confusion_matrix(y_test, rfc_classifier.predict(X_test))
    print('RFC', i, rfc_classifier.score(X_test, y_test), cm[0, 1] + cm[1, 0])
    precision = cm[1, 1]/(cm[1, 1]+cm[0, 1])
    recall = cm[1, 1]/(cm[1, 1]+cm[1, 0])
    print(precision, recall, 'n', 'F1', 2*precision*recall/(precision+recall))

    
#%%
from sklearn.ensemble import GradientBoostingClassifier
for i in range(100, 1000, 100):
    gbm_classifier = GradientBoostingClassifier(n_estimators = i)
    gbm_classifier.fit(X_train, y_train)
    cm = confusion_matrix(y_test, gbm_classifier.predict(X_test))
    print('GBM', i, gbm_classifier.score(X_test, y_test), cm[0, 1] + cm[1, 0])
    precision = cm[1, 1]/(cm[1, 1]+cm[0, 1])
    recall = cm[1, 1]/(cm[1, 1]+cm[1, 0])
    print(precision, recall, 'n', 'F1', 2*precision*recall/(precision+recall))
