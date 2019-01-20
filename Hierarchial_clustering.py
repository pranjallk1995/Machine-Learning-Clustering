#Hierarchial Clustering

#importing libraries.
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch

#importing data
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

#using dendogram to get the optimal number of clusters.
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    
#plotting dendogram.
plt.title("Dendogram")
plt.xlabel("Customers")
plt.ylabel("Euclidian Distances")
plt.show()

#applying Hierarchial Clustering.
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
pred = ac.fit_predict(X)

#Visualization.
plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s = 20, c = 'red', label = 'Cluster 1', marker = 'x')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s = 20, c = 'blue', label = 'Cluster 2',  marker = 'x')
plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s = 20, c = 'green', label = 'Cluster 3',  marker = 'x')
plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4',  marker = 'x')
plt.scatter(X[pred == 4, 0], X[pred == 4, 1], s = 20, c = 'orange', label = 'Cluster 5',  marker = 'x')
plt.title("Mall Customer Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()