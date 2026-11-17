import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from time import time

df_diamonds = sns.load_dataset('diamonds')
DATA = df_diamonds[['carat', 'depth', 'table', 'price', 'x', 'y', 'z']]

def kmeans(X, k):

    km = KMeans(n_clusters=k)
    km.fit(X)

    labels = km.labels_
    centroids = km.cluster_centers_

    return (centroids, labels)

def kmeans_diamonds(n, k):

    first_n = DATA[:n]

    return kmeans(first_n, k)

def kmeans_timer(n, k, n_iter=5):
    runs = []
    for i in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        t = time() - start
        runs.append(t)

    average = np.mean(runs)

    return average
