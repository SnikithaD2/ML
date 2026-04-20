# Question: K-Means Clustering for Mall Customers
#
# Description:
# K-Means is an unsupervised learning algorithm used for clustering data into groups
# based on similarity. It partitions data into K clusters by minimizing the distance
# between data points and cluster centroids. The algorithm iteratively updates centroids
# until convergence. It is widely used in customer segmentation, pattern recognition,
# and market analysis. Choosing the correct number of clusters is important for meaningful
# results and accurate grouping of similar data points.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("mallcustomers.csv")

X = df[['Income', 'SpendingScore']]

kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

print(df[['Income', 'SpendingScore', 'Cluster']].head(20))

plt.scatter(X['Income'], X['SpendingScore'], c=df['Cluster'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.show()
