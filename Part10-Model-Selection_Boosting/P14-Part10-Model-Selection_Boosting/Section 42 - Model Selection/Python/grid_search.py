# Grid Search

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values 
y = dataset.iloc[:,4].values

#splitting the dataset into test and trainninng set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Fitting kernel svm into train set
from sklearn.svm import SVC 
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

#predicting the test set result
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Applyin k-fold Cross validaton
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X = X_train, y = y_train,cv=10)
print('accuracy is of k-fold:',accuracies.mean())
print('standardDeviation is of k-fold:',accuracies.std())

#Applying grid search to find the bestparametes
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1,10,100,1000],'kernel':['linear']},
              {'C':[1,10,100,100],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
              ] #SVC parameters WE are checking if our model is linear and non linear by providing key value pair
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train,y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set,y_set = X_train,y_train
X1,X2 = np.meshgrid(np.arange(X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step=0.01),
                    np.arange(X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.01)
                    )
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
                c = ['blue','black'][i],label = j)

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
                c = ['red','green'][i],label=j)
    
plt.title('Logistic Regression(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

