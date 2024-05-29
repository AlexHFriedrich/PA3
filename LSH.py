import time
import numpy as np
from sklearn.metrics import pairwise_distances
from KMeans import KMeans
from tqdm import tqdm




class LloydsAlgorithmLSH(KMeans):
    def __init__(self, k, data, true_labels, num_hash_tables=5, num_hash_functions=10, w=1.0):
        super().__init__(k, data, true_labels)
        self.distances = None
        self.converged = False
        self.time = 0
        self.num_distance_calculations = 0
        self.lsh = LSH(data, num_hash_tables, num_hash_functions, w)
        self.lsh.build_hash_tables()

    def fit(self):
        pass

    def initialize_centroids(self):
        return self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]

    def update_centroids(self):
        for i in range(self.k):
            self.centroids[i] = np.mean(self.clusters[i], axis=0)

    def assign_clusters(self):
        return None


class LSH:
    def __init__(self, data, num_hash_tables, num_hash_functions, w=1.0):
        self.data = data
        self.num_hash_tables = num_hash_tables
        self.num_hash_functions = num_hash_functions
        self.w = w
        self.hash_tables = [{} for _ in range(num_hash_tables)]
        self.hash_functions = self.generate_hash_functions()

    def generate_hash_functions(self):
        hash_functions = []
        for _ in range(self.num_hash_tables):
            table_functions = []
            for _ in range(self.num_hash_functions):
                a = np.random.randn(self.data.shape[1])
                b = np.random.uniform(0, self.w)
                table_functions.append((a, b))
            hash_functions.append(table_functions)
        return hash_functions

    def hash(self, vector, hash_functions):
        hash_value = tuple(int((np.dot(vector, a) + b) / self.w) for a, b in hash_functions)
        return hash_value

    def build_hash_tables(self):
        for idx, vector in enumerate(self.data):
            for table_idx, hash_functions in enumerate(self.hash_functions):
                hash_value = self.hash(vector, hash_functions)
                if hash_value not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_value] = []
                self.hash_tables[table_idx][hash_value].append(idx)

    def query(self, vector):
        candidates = set()
        for table_idx, hash_functions in enumerate(self.hash_functions):
            hash_value = self.hash(vector, hash_functions)
            if hash_value in self.hash_tables[table_idx]:
                candidates.update(self.hash_tables[table_idx][hash_value])
        return list(candidates)