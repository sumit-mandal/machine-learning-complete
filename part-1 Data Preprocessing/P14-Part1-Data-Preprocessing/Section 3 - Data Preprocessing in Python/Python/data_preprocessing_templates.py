# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values #it means we take all rows and all columnns except last
Y = dataset.iloc[:,3].values #it shows all rows and only 3 column

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3]) #used for fitting imputer object to matrix X
X[:, 1:3] = imputer.transform(X[:, 1:3]) #X[:, 1:3] variable is used to take columns for missing data
#imputer.transform(X[:, 1:3]) is method which is used to replace missing values with mean.
print(X)


#Encoding categorial data
from sklearn.preprocessing import LabelEncoder