import gc
from collections import defaultdict
import numpy as np
import time
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from LloydsAlgorithm import LloydsAlgorithm


class LloydsAlgorithmLSH(LloydsAlgorithm):
    def __init__(self, k, data, true_labels, num_hash_tables=2, num_hashes_per_table=3, bucket_size=1.0, max_iter=100,
                 debug=False, tol=2*1e-2):
        super().__init__(k, data, true_labels, max_iter, tol=tol)
        self.points_assigned_by_distance = None
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

    def _initialize_centroids(self):
        return self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]

    def _assign_clusters(self):
        temp_clusters = self.clusters
        self.clusters = {k: [] for k in range(self.k)}
        assigned_points = set()

        # First assign points from hash buckets
        for table_index, hash_table in enumerate(self.centroid_hash_buckets):
            for hash_key, centroid_indices in hash_table.items():
                data_point_indices = self.hash_buckets[table_index].get(hash_key, [])
                if len(centroid_indices) > 0:
                    first_centroid_index = centroid_indices[0]
                    for data_point_index in data_point_indices:
                        if data_point_index not in assigned_points:
                            self.clusters[first_centroid_index].append(data_point_index)
                            self.labels[data_point_index] = first_centroid_index
                            assigned_points.add(data_point_index)
        if self.debug:
            print(f"Assigned points from hash buckets: {len(assigned_points)} out of {len(self.data)}")

        # Handle remaining points by distance
        remaining_points = list(set(range(self.data.shape[0])) - assigned_points)
        if remaining_points:
            self.distances = pairwise_distances(self.data[remaining_points], self.centroids)
            for i, idx in enumerate(remaining_points):
                cluster = np.argmin(self.distances[i])
                self.labels[idx] = int(cluster)
                self.clusters[cluster].append(idx)
                assigned_points.add(idx)
        if self.debug:
            print(f"Total assigned points after handling remaining buckets: {len(remaining_points)}")

        if self.n_iter_ > 1:
            return self._convergence_check(temp_clusters), len(remaining_points)
        else:
            return False, len(remaining_points)

    def fit(self):
        start = time.time()
        for _ in tqdm(range(self.max_iter)):
            self._hash_centroids()
            self.converged, self.points_assigned_by_distance = self._assign_clusters()
            conv = self._step()
            if conv:
                break
            self.num_distance_calculations += self.points_assigned_by_distance * self.k
        self.NMI = self._NMI()
        self.time = time.time() - start

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
