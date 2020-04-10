#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:55:36 2020

@author: admangli
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#%%
# Convert dataset into list of strings list
transactions = []

for i in range(len(dataset)):
    transaction = dataset.iloc[i].values
    temp = []
    for j in transaction:
        if str(j) != 'nan':
            temp.append(str(j))
    transactions.append(temp)
    
#%%

def check(list1, list2):
    # Check if list 1 is contained in list 2
    for x in list1:
        if (x in list2) == False:
            return False
    return True

# Define minimum support
min_support = 0.01

#%%
# Eclat implementation

# Obtain all product groups with more than minimum support
    # Creating bag of words dictionary for all possible products in the transactions list
product_dict = dict()
dict_indexer = 0
for i in transactions:
    for j in i:
        if product_dict.get(j) == None:
            product_dict[j] = dict_indexer
            dict_indexer += 1
            
    # Convert all transactions to product serials for faster processing
serialized_transactions = []
for i in transactions:
    serialized_products = []
    for j in i:
        serialized_products.append(product_dict[j])
    serialized_transactions.append(serialized_products)
    
    # Create sets of 2 from product list and calculate support for each from transactions
    # TODO Extend for groups of size > 2
product_groups = []
for i in range(len(product_dict)):
    j = i + 1
    while j < len(product_dict):
        product_groups.append([i, j])
        j += 1
#%%
    # Calculate support of each product group in transactions
    # Support  = Data Samples / Population Size
support_counters = []
for prod_gp in product_groups:
    support_count = 0
    for transaction in serialized_transactions:
        if check(prod_gp, transaction):
            support_count += 1
    support_counters.append(support_count/len(serialized_transactions))
#%%
# Sort associations by decreasing support
associated_products = []
inverse_product_dict = dict(zip(product_dict.values(), product_dict.keys()))
#%%
for idx, value in enumerate(support_counters):
    # Convert product groups into named representations
    if value > min_support:
        associated_products.append([value, inverse_product_dict[product_groups[idx][0]], inverse_product_dict[product_groups[idx][1]]])
associated_products = sorted(associated_products, reverse = True)