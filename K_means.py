#K-Means Clustering

#importing libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

#using elbow method to get the optimal number of clusters.
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
#plotting the elbow graph.
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#applying K-means.
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
pred = kmeans.fit_predict(X)

#Visualizing K-means.
plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s = 20, c = 'red', label = 'Cluster 1', marker = 'x')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s = 20, c = 'blue', label = 'Cluster 2',  marker = 'x')
plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s = 20, c = 'green', label = 'Cluster 3',  marker = 'x')
plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4',  marker = 'x')
plt.scatter(X[pred == 4, 0], X[pred == 4, 1], s = 20, c = 'orange', label = 'Cluster 5',  marker = 'x')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'black', label = 'Centroid')
plt.title("Mall Customer Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()