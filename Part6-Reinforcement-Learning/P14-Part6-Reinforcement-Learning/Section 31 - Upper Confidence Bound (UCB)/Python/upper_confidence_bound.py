# Upper Confidence Bound (UCB)

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#immportin the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB
import math
N = 10000
d = 10 #no of ads
ads_selected = [] #it'll give us the list about all the different version of the ads that are selected at each round

numbers_of_selections = [0] * d #it creates vector of size d containg only zero
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0,N): #it'll loop over rows
    ad = 0
    max_upper_bound = 0
    for i in range(0,d): #it'll loop over columns
        if (numbers_of_selections[i]>0):
            average_rewards = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/ numbers_of_selections[i])
            upper_bound = average_rewards+delta_i
        else:
            upper_bound = 1e400 #(10^400)
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i  #it keeps track of the index which has max_upper_bound
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
            
            
"""#we don't know which ad have highest click through rate so for the first 10 rounds we will simply select 10 first ads
#i.e. round 1 will select ad 1 ,round 2 will select ad 2 and so on until round 10
#it will also give rough idea of sums_of_rewards ass it will give reward 0 if ads were not clicked and 1 if ads were clicked"""
#that is why we used if (number_of_selections[i]>0):
            # average_rewards = sums_of_rewards[i]/number_of_selections[i]
            # delta_i = math.sqrt(3/2 * math.log(n + 1)/ number_of_selections[i])
            # upper_bound = average_rewards+delta_i
            
#Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
            
            
            
           
