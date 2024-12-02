import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import DistanceMetric
from mpi4py import MPI
from datetime import datetime
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

groups = 4
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=1410)
dist_metric = DistanceMetric.get_metric('euclidean')

start = datetime.utcnow()

# Split data for all ranks
if rank == 0:
    centroids = X[np.random.choice(X.shape[0], size=groups, replace=False)]
    ave, res = divmod(X.shape[0], size)
    counts = [ave + 1 if p < res else ave for p in range(size)]
    starts = [sum(counts[:p]) for p in range(size)]
    ends = [sum(counts[:p + 1]) for p in range(size)]
    X_split = [X[starts[p]:ends[p]] for p in range(size)]
else:
    X_split = None
    centroids = None

# Send work data to all ranks
X_local = comm.scatter(X_split, root=0)
centroids = comm.bcast(centroids, root=0)

while True:
    distances = dist_metric.pairwise(X_local, centroids)
    local_clusters = np.argmin(distances, axis=1)

    # Update locally from partial data
    new_centroids = np.zeros_like(centroids)
    counts = np.zeros(groups)
    for i in range(groups):
        cluster_assign = X_local[local_clusters == i]
        if len(cluster_assign) > 0:
            new_centroids[i] = np.mean(cluster_assign, axis=0)
            counts[i] = len(cluster_assign)

    # Gather all data after iteration at central rank
    partial_centroids = comm.gather(new_centroids, root=0)
    partial_counts = comm.gather(counts, root=0)

    if rank == 0:
        full_centroids = np.zeros_like(centroids)
        total_counts = np.zeros(groups)

        # Calculate centroids based on partial results
        for centroid, counts in zip(partial_centroids, partial_counts):
            full_centroids += centroid * counts[:, None]
            total_counts += counts
        full_centroids = full_centroids / total_counts[:, None]
        should_break = True if np.allclose(centroids, full_centroids) else False
    else:
        full_centroids = None
        should_break = False

    # Release control
    centroids = comm.bcast(full_centroids, root=0)
    should_break = comm.bcast(should_break, root=0)

    if should_break:
        break

local_assign = np.array([[i, cluster] for i, cluster in zip(range(len(X_local)), local_clusters)])

# Final assembly
partial_assign = comm.gather(local_assign, root=0)
if rank == 0:
    final_clusters = np.zeros(X.shape[0], dtype=int)
    offset = 0
    for p, local_assignment in enumerate(partial_assign):
        for local_id, cluster in local_assignment:
            global_id = offset + local_id
            final_clusters[global_id] = cluster
        offset += len(partial_assign[p])

    print(f'Processing time: {datetime.utcnow() - start}[s]')
    plt.scatter(X[:, 0], X[:, 1], c=final_clusters, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', edgecolor="k", s=80)
    plt.show()
