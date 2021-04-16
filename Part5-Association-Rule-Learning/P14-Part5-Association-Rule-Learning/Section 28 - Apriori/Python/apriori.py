# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv",header=None)

transactions=[]
for i in range(0,7501): #i will take all index from 0,7501
    # print(i)
#1st loop - looping over all the transaction in dataset
#2nd loop - looping over all the product in each of transaction
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# for each transaction corresponding to each index of i here we want to create
# a list of the different products of this transaction.
# append function will add each of the different transaction to this whole list of transactions.
#transaction.append[i,j] for j in range(0,20)] we added all the value of i but we also need to add j, 
# we do this by looping over all the column i.e. using range(0,20).
# we add square bracket[] before str and after (0,20) as we want each transaction to be a list that is 
# we want each transaction to contain all the different products within the list.
# apriori function is expecting the different products and the different transactions as strings. so we add str before dataset.values


#Training Apriori to the  dataset
from apyori import apriori
rules = apriori(transactions,min_support=0.003,min_confidence = 0.2, min_lift = 3 ,min_length = 2)
#min support = items which are purchased 3 times per day. i.e. 7*3 = 21 times a week. (7*3)/7500=0.0028. 7500 = total number of items

#visualising the results
results = list(rules)


for i in results:
    print(i)
    print(' **** ')