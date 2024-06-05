import gc
from collections import defaultdict

import numpy as np
import time
from sklearn.metrics import pairwise_distances
from KMeans import KMeans
from tqdm import tqdm

from LloydsAlgorithm import LloydsAlgorithm


class LloydsAlgorithmLSH(LloydsAlgorithm):
    def __init__(self, k, data, true_labels, num_hash_tables=2, num_hashes_per_table=3, bucket_size=1.0, max_iter=100,
                 debug=False):
        super().__init__(k, data, true_labels, max_iter)
        self.num_assignments = None
        self.distances = None
        self.converged = False
        self.time = 0
        self.num_distance_calculations = 0
        self.num_hash_tables = num_hash_tables
        self.num_hashes_per_table = num_hashes_per_table
        self.bucket_size = bucket_size
        self.hash_tables = self._generate_hash_tables()
        self.hash_buckets = self._hash_data()
        self.centroid_hash_buckets = [None] * self.num_hash_tables
        self.debug = debug

    def _generate_hash_functions(self):
        hash_functions = []
        for _ in range(self.num_hashes_per_table):
            a = np.random.normal(size=self.data.shape[1])
            b = np.random.uniform(0, self.bucket_size)
            hash_functions.append((a, b))
        return hash_functions

    def _generate_hash_tables(self):
        hash_tables = []
        for _ in range(self.num_hash_tables):
            hash_tables.append(self._generate_hash_functions())
        return hash_tables

    def _hash_point(self, point, hash_functions):
        hash_values = []
        for a, b in hash_functions:
            hash_value = np.floor((np.dot(point, a) + b) / self.bucket_size)
            hash_values.append(hash_value)
        return tuple(hash_values)

    def _hash_data(self):
        hash_buckets = [{} for _ in range(self.num_hash_tables)]
        for i, point in enumerate(self.data):
            for j, hash_functions in enumerate(self.hash_tables):
                hash_values = self._hash_point(point, hash_functions)
                if hash_values not in hash_buckets[j]:
                    hash_buckets[j][hash_values] = []
                hash_buckets[j][hash_values].append(i)
        return hash_buckets

    def _hash_centroids(self):
        self.centroid_hash_buckets = [{} for _ in range(self.num_hash_tables)]
        for j, hash_functions in enumerate(self.hash_tables):
            for i, centroid in enumerate(self.centroids):
                hash_values = self._hash_point(centroid, hash_functions)
                if hash_values not in self.centroid_hash_buckets[j]:
                    self.centroid_hash_buckets[j][hash_values] = []
                self.centroid_hash_buckets[j][hash_values].append(i)

    def initialize_centroids(self):
        return self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]

    def update_centroids(self):
        for i in range(self.k):
            if len(self.clusters[i]) > 0:
                self.centroids[i] = np.mean(self.clusters[i], axis=0)
            else:
                # Randomly reinitialize centroid if cluster is empty
                self.centroids[i] = self.data[np.random.choice(self.data.shape[0])]
                if self.debug:
                    print(f"Cluster {i} is empty. Reinitializing centroid to {self.centroids[i]}")

    def assign_clusters(self):
        temp_clusters = self.clusters
        self.clusters = {k: [] for k in range(self.k)}
        assigned_points = set()

        # First assign points from hash buckets
        for j in range(self.num_hash_tables):
            for bucket, centroids in self.centroid_hash_buckets[j].items():
                if bucket in self.hash_buckets[j]:
                    for idx in self.hash_buckets[j][bucket]:
                        if idx not in assigned_points:
                            cluster = centroids[np.random.choice(len(centroids))]
                            self.labels[idx] = int(cluster)
                            self.clusters[cluster].append(self.data[idx])
                            assigned_points.add(idx)
        if self.debug:
            print(f"Assigned points from hash buckets: {len(assigned_points)} out of {len(self.data)}")

        # Handle remaining points by distance
        remaining_points = list(set(range(self.data.shape[0])) - assigned_points)
        if remaining_points:
            self.distances = pairwise_distances(self.data[remaining_points], self.centroids)
            for i, idx in enumerate(remaining_points):
                cluster = np.argmin(self.distances[i])
                self.labels[idx] = int(cluster)
                self.clusters[cluster].append(self.data[idx])
                assigned_points.add(idx)
        if self.debug:
            print(f"Total assigned points after handling remaining buckets: {len(assigned_points)}")

        if self.n_iter_ > 1:
            return self.convergence_check(temp_clusters), len(remaining_points)
        else:
            return False, len(remaining_points)

    def fit(self):
        start = time.time()
        for _ in tqdm(range(self.max_iter)):
            self._hash_centroids()
            self.converged, self.num_assignments = self.assign_clusters()
            if self.converged:
                print('Converged after {} iterations'.format(self.n_iter_))
                break
            self.update_centroids()
            self.losses.append(self.compute_loss())
            self.n_iter_ += 1
            self.num_distance_calculations += (self.data.shape[0] - self.num_assignments) * self.k
        self.NMI = self._NMI()
        self.time = time.time() - start

    def compute_loss(self):
        loss = 0
        for i in range(self.k):
            if len(self.clusters[i]) > 0:
                loss += np.sum(pairwise_distances(self.clusters[i], [self.centroids[i]]))
        return loss / self.data.shape[0]

    def convergence_check(self, temp_clusters):
        for i in range(self.k):
            if not np.array_equal(temp_clusters[i], self.clusters[i]):
                return False
        return True

    def grid_search_LSH(self, grid):
        results = defaultdict(list)
        for num_hash_tables in grid["num_hash_tables"]:
            for num_hashes_per_table in grid["num_hashes_per_table"]:
                if abs(num_hash_tables - num_hashes_per_table) < 3:
                    for bucket_size in grid["bucket_size"]:
                        NMI_lsh = []
                        runtimes_lsh = []
                        num_assigned_values = []

                        for _ in range(3):
                            lloyds_lsh = LloydsAlgorithmLSH(self.k, self.data.copy(), self.true_labels,
                                                            num_hash_tables=num_hash_tables,
                                                            num_hashes_per_table=num_hashes_per_table,
                                                            bucket_size=bucket_size,
                                                            max_iter=self.n_iter_,
                                                            debug=False)
                            lloyds_lsh.fit()
                            NMI_lsh.append(lloyds_lsh.NMI)
                            runtimes_lsh.append(lloyds_lsh.time)
                            num_assigned_values.append(lloyds_lsh.num_assignments)

                        NMI = sum(NMI_lsh) / len(NMI_lsh)
                        config_str = f"nht={num_hash_tables}, nhpt={num_hashes_per_table}, bs={bucket_size}"
                        results[config_str] = [NMI, sum(runtimes_lsh) / len(runtimes_lsh),
                                               sum(num_assigned_values) / len(num_assigned_values)]
                        gc.collect()

        for key, value in results.items():
            print(f"{key}: {value}")

        with open('results/results_hyperparameter_grid_search_lsh.txt', 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
