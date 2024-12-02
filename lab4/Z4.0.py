import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import DistanceMetric
from datetime import datetime
import matplotlib.pyplot as plt

NUM_GROUPS = 4

# Datasets creation with 2 informative features
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=1410)

start = datetime.utcnow()

centroids = X[np.random.choice(X.shape[0], size=NUM_GROUPS, replace=False)]
dist_metric = DistanceMetric.get_metric('euclidean')

prev_centroids = np.zeros(np.shape(centroids))
clusters = np.zeros(X.shape[0])
dist_table = np.zeros((X.shape[0], NUM_GROUPS))

while True:
    dist_table = dist_metric.pairwise(X, centroids)
    cluster_assigned = np.argmin(dist_table, axis=1)
    prev_centroids = np.copy(centroids)

    for i in range(NUM_GROUPS):
        centroids[i] = np.mean(X[cluster_assigned == i], axis=0)

    if np.allclose(prev_centroids, centroids):
        break

print(f'Processing time: {datetime.utcnow() - start}[s]')
plt.scatter(X[:, 0], X[:, 1], c=cluster_assigned, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', edgecolor="k", s=80)
plt.show()
