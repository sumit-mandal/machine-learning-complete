# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("mall_customers.csv")
X = dataset.iloc[:,[3,4]].values

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = [] #Within-Cluster Sum of Square

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300,n_init=10,random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #kmeans.inertia_ computes he wcss. later we append it with wcss

#Applying kmeans to the mall dataset
kmeans = KMeans(n_clusters = 5,init = 'k-means++',max_iter = 300,n_init = 10,random_state=0)
y_kmeans = kmeans.fit_predict(X)
#for every single client of our dataset the fit_predict is going to tell the cluster to Which client belongs.
#and it'll return it's cluster numbers into a single vector that we'll call y_kmeans

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
#[y_kmeans == 0] meaning we want the observation that belongs to cluster 1 [y_kmeans == 0,0] meaning we want 1st column of our data X
#X[y_kmeans == 0, 0] by doing this we gave x coordinates of all the observation points that belongs to cluster 1. 
#X[y_kmeans == 0, 1] here by changing 0 to 1 our dataset corresponds to the second column of our data X that is the y coodinate
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
#it is same as above but is use to plot centeroid
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()