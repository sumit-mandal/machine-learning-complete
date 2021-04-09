# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Fitting Logistic regression to the training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)
#It helps classifier to learn co-relation between x_train and y_train 

#predicting the test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#it evaluates if our logistic regression model learnt and understood correctly the corelation in training sets
# and see if it can makes powerful prediction.


#Visualising the training set results
from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max() + 1, step=0.01 ),#it is for age
                    np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max() + 1, step=0.01)
                    ) 
#We take min-1 and max+1 so that point don't gets squeezed thus making minimumand maximumum value to be -2 and 3 respectively
#we prepare the grid with all the pixel points
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap = ListedColormap(('red','green')))
#it colorizes all the pixel. and draws the contour line between those who bought and didn't buy
#predict function is use to predict wether each classifier belongs to class 0 or class 1. If pixel point belongs to class 0 it will be colorized red and
#if it belongs to 1 it will be colorized green.
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
 #sets the limit for current axes
 
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                c = ListedColormap(('blue','black'))(i),label = j)
#with this loop we plot all the datapoints.
plt.title('Logistic Regression(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#visualising the test set result
from matplotlib.colors import ListedColormap
X_set,y_set = X_test,y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max() + 1,step = 0.01),
                    np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max() + 1,step = 0.01)
                    )
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap = ListedColormap(('blue','black')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set== j,1],
                c = ListedColormap(('red','green'))(i),label=j)
    
plt.title('Logistic Regression(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
