# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)

#importing the datset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values
print(X)
print(y)



#Encoding categorical data(Converting string into numbers) here georaphy and genders are converted
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
 #it creates dummy variable
 
 #feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)


#splitting dataset into test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#part2 - Making the ANN

#importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense


#initialising the classifier
#defining sequential i.e sequense of layers


classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6,activation = 'relu'))
#units = 6 as no. of column in X_train = 11 and y_train =1 --> 11+1/2

#Adding the second hidden lyer
classifier.add(Dense(units = 6, activation='relu'))

#adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#compiling the ann
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ann into trainninng set
classifier.fit(X_train,y_train,batch_size=32,epochs=100)
#batch_size is numbr of observations after which you want to update the weights

#predicting the test set
y_pred = classifier.predict(X_test)
y_pred  = (y_pred>0.5) #it converts value into true and false. It return value true if customer have more than 50% chance of leaving



#making thhe confusion matrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
