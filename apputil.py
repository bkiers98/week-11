import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from time import time


df_diamonds = sns.load_dataset('diamonds')
DATA = df_diamonds[[
    'carat',
    'depth',
    'table',
    'price',
    'x',
    'y',
    'z'
    ]]

def kmeans(X, k):
    """
    This function performs k-means clustering on a numerical
    NumPy array X and returns a tuple (centroids, labels).
    """
    km = KMeans(n_clusters=k)
    km.fit(X)

    labels = km.labels_
    centroids = km.cluster_centers_

    return (centroids, labels)

def kmeans_diamonds(n, k):
    """
    This function calls kmeans() on a number of rows n from the
    numerical columns of the diamonds dataset.
    """
    first_n = DATA[:n]

    return kmeans(first_n, k)

def kmeans_timer(n, k, n_iter=5):
    """
    This function returns the average time it takes to run
    kmeans_diamonds when run n_iter number of times. 
    """
    runs = []
    for i in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        t = time() - start
        runs.append(t)

    average = np.mean(runs)

    return average
