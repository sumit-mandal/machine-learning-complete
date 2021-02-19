# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values #independent vriables
y = dataset.iloc[:,1].values #dependent variables

#on looking output we find X is matrix as we have 1 column whereas in y we have no column therefore it is vector

#splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test =  train_test_split(X, y, test_size = 1/3, random_state = 0)

#X_train is matrix of indepenndent variable and y_train is dependent variable factor for trainig set
#X_test is matrix of indepenndent variable and y_test is dependent variable factor for test set

#fitting Simple Linear Regression into the training set