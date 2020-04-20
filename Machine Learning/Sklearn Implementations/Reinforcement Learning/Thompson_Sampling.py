import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv').values

#%%
slot_machines = 10

#%% Random ad selection reward
import random

random_reward = 0

for i in range(len(dataset)):
    random_reward += dataset[i, random.randint(0, slot_machines - 1)]
    
#%%
number_of_positive_reward = [0]*slot_machines
number_of_negative_reward = [0]*slot_machines    
thompson_reward = 0
ad_selection_sequence = []

for round in range(slot_machines, len(dataset)):
    selected_ad = 0 # Ad to be selected
    max_random = 0 # Maximum random draw from all expected distributions
    
    for i in range(0, slot_machines):
        random_beta = random.betavariate( number_of_positive_reward[i]+1, 
                                          number_of_negative_reward[i]+1 )
        if random_beta>max_random:
            max_random = random_beta
            selected_ad = i

    reward = dataset[round][selected_ad]
    if reward == 1:
        number_of_positive_reward[selected_ad] += 1
    else:
        number_of_negative_reward[selected_ad] += 1
    
    ad_selection_sequence.append(selected_ad)
    thompson_reward += reward
        
#%% Visualize results

# Plot a histogram showing how many times each ad was selected
plt.hist(ad_selection_sequence)
plt.xlabel('Ad Number')
plt.ylabel('Number of selections')
plt.title('Ad selection comparision')
plt.show()