# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
#linkage is algorith itself of hiererichal clustering
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#fitting hierarichal clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,affinity='euclidean',linkage='ward')
#affininty = distance to linkage
y_hc = hc.fit_predict(X)

#Visualising the result
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s=100,c = 'red' ,label='Careful')
# y_hc == 0 selects those elements where y_hc[i] is equal to 0. 
# X[y_hc == 0, 0] selects the elements of X where the corresponding y_kmeans value is
# 0 and the second dimension is 0.
plt.scatter(X[y_hc ==1,0],X[y_hc == 1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_hc ==2,0],X[y_hc==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_hc == 3,0],X[y_hc==3,1],s=100,c='cyan',label='careless')
plt.scatter(X[y_hc == 4,0],X[y_hc==4,1],s=100,c='magenta',label='sensible')
plt.tilte('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()


