import time

import numpy as np
from sklearn.metrics import pairwise_distances
from KMeans import KMeans
from tqdm import tqdm


class LloydsAlgorithm(KMeans):
    def __init__(self, k, data, true_labels, max_iter):
        super().__init__(k, data, true_labels, max_iter)
        self.distances = None
        self.converged = False
        self.time = 0
        self.num_distance_calculations = 0

    def initialize_centroids(self):
        return self.data[:self.k]

    def update_centroids(self):
        for i in range(self.k):
            self.centroids[i] = np.mean(self.clusters[i], axis=0)

    def assign_clusters(self):
        self.distances = self.calculate_distance()

        temp_clusters = self.clusters
        self.clusters = {k: [] for k in range(self.k)}

        for i in range(self.data.shape[0]):
            cluster = np.argmin(self.distances[i])
            self.labels[i] = int(cluster)
            self.clusters[cluster].append(self.data[i])

        if self.iterations > 1:
            return self.convergence_check(temp_clusters)
        else:
            return False

    def calculate_distance(self):
        return pairwise_distances(self.data, self.centroids)

    def fit(self):
        start = time.time()
        for _ in tqdm(range(self.max_iter)):
            self.converged = self.assign_clusters()
            if self.converged:
                print('Converged after {} iterations'.format(self.iterations))
                break
            self.update_centroids()
            self.losses.append(self.compute_loss())
            self.iterations += 1
            self.num_distance_calculations += self.data.shape[0] * self.k
        # self.convergence_plot()
        self.NMI = self._NMI()
        self.time = time.time() - start

    def convergence_check(self, temp_clusters):
        for i in range(self.k):
            if not np.array_equal(temp_clusters[i], self.clusters[i]):
                return False
        return True
