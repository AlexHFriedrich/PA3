import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances, normalized_mutual_info_score
from KMeans import KMeans
from tqdm import tqdm


class LloydsAlgorithm():
    def __init__(self, k, data, true_labels, max_iter, random_init=False):
        self.random_init = random_init
        self.k = k
        self.data = data
        self.n_iter_ = 0
        self.losses = []
        self.NMI = 0
        self.true_labels = self._update_true_labels(true_labels)
        self.clusters = dict()
        self.centroids = self.initialize_centroids()
        self.labels = np.zeros(self.data.shape[0])
        self.max_iter = max_iter
        self.distances = None
        self.converged = False
        self.time = 0
        self.num_distance_calculations = 0

    def initialize_centroids(self):
        if self.random_init:
            return self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]
        return self.data[:self.k]

    def update_centroids(self):
        for i in range(self.k):
            if len(self.clusters[i]) == 0:
                self.centroids[i] = self.data[np.random.choice(self.data.shape[0])]
            else:
                self.centroids[i] = np.mean(self.clusters[i], axis=0)

    def assign_clusters(self):
        self.distances = self.calculate_distance()

        temp_clusters = self.clusters
        self.clusters = {k: [] for k in range(self.k)}

        for i in range(self.data.shape[0]):
            cluster = np.argmin(self.distances[i])
            self.labels[i] = int(cluster)
            self.clusters[cluster].append(self.data[i])

        if self.n_iter_ > 1:
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
                print('Converged after {} iterations'.format(self.n_iter_))
                break
            self.update_centroids()
            self.losses.append(self.compute_loss())
            self.n_iter_ += 1
            self.num_distance_calculations += self.data.shape[0] * self.k
        # self.convergence_plot()
        self.NMI = self._NMI()
        self.time = time.time() - start

    def convergence_check(self, temp_clusters):
        for i in range(self.k):
            if not np.array_equal(temp_clusters[i], self.clusters[i]):
                return False
        return True

    def predict(self, data):
        self.num_distance_calculations += data.shape[0] * self.k
        return np.argmin(pairwise_distances(data, self.centroids), axis=1)

    def compute_loss(self):
        loss = 0
        for i in range(self.k):
            if len(self.clusters[i]) > 0 and len(self.centroids[i]) > 0:
                loss += np.sum([np.linalg.norm(self.clusters[i] - self.centroids[i]) ** 2])
        return loss / self.data.shape[0]

    def convergence_plot(self):
        plt.plot(self.losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Convergence of Lloyd\'s Algorithm')
        plt.savefig('lloyds_algorithm_convergence.png')
        plt.show()

    def _NMI(self):
        return normalized_mutual_info_score(self.true_labels, self.labels)

    @staticmethod
    def _update_true_labels(true_labels):
        unique_labels = list(set(true_labels))
        true_label_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
        return [true_label_dict[label] for label in true_labels]
