import time
import numpy as np
from sklearn.metrics import pairwise_distances, normalized_mutual_info_score
from tqdm import tqdm


class LloydsAlgorithm:
    def __init__(self, k, data, true_labels, max_iter, random_init=False, tol=1e-2):
        self.tol = tol
        self.random_init = random_init
        self.k = k
        self.data = data
        self.n_iter_ = 0
        self.losses = []
        self.NMI = 0
        self.true_labels = true_labels
        self.clusters = dict()
        self.centroids = self._initialize_centroids()
        self.labels = np.zeros(self.data.shape[0])
        self.max_iter = max_iter
        self.distances = None
        self.converged = False
        self.time = 0
        self.num_distance_calculations = 0

    def _initialize_centroids(self):
        """
        Initialize the centroids of the clusters either as k random data points or the first k data points
        :return: initial centroids
        """
        if self.random_init:
            return self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]
        return self.data[:self.k]

    def _update_centroids(self):
        """
        Update the centroids of the clusters to random data points if cluster is empty, otherwise update to the mean
        of the current cluster.
        """
        change_in_centroids = 0
        for i in range(self.k):
            if len(self.clusters[i]) == 0:
                change_in_centroids += 1e4
                self.centroids[i] = self.data[np.random.choice(self.data.shape[0])]
            else:
                temp_centroid = self.centroids[i].copy()
                self.centroids[i] = np.mean(self.clusters[i], axis=0)
                change_in_centroids += np.linalg.norm(temp_centroid - self.centroids[i])
        return change_in_centroids/len(self.centroids)

    def _assign_clusters(self):
        """
        assign each data point to the closest centroid and check if the clusters have converged
        """
        self.distances = self._calculate_distance()

        temp_clusters = self.clusters
        self.clusters = {k: [] for k in range(self.k)}

        for i in range(self.data.shape[0]):
            cluster = np.argmin(self.distances[i])
            self.labels[i] = int(cluster)
            self.clusters[cluster].append(self.data[i])

        if self.n_iter_ > 1:
            return self._convergence_check(temp_clusters)
        else:
            return False

    def _calculate_distance(self):
        """
        Calculate the n by k distance matrix between the data points and the centroids
        :return: distance matrix
        """
        return pairwise_distances(self.data, self.centroids)

    def fit(self):
        """
        Fit the model to the data
        :return:
        """
        start = time.time()
        for _ in tqdm(range(self.max_iter)):
            self.converged = self._assign_clusters()
            relative_change_in_centroids = self._update_centroids()

            if self.converged or (self.n_iter_ > 10 and relative_change_in_centroids < self.tol):
                print('Converged after {} iterations'.format(self.n_iter_))
                break
            self.losses.append(self._compute_loss())
            self.n_iter_ += 1
            self.num_distance_calculations += self.data.shape[0] * self.k
        # self.convergence_plot()
        self.NMI = self._NMI()
        self.time = time.time() - start

    def _convergence_check(self, temp_clusters):
        """
        Check if the clusters of the previous iteration are the same as the current iteration
        :param temp_clusters: clusters from the previous iteration
        :return: boolean indicating if the clusters have converged
        """
        for i in range(self.k):
            if not np.array_equal(temp_clusters[i], self.clusters[i]):
                return False
        return True

    def predict(self, data):
        """
        Predict the clusters for each data point using the current centroids
        :param data:
        :return:
        """
        self.num_distance_calculations += data.shape[0] * self.k
        return np.argmin(pairwise_distances(data, self.centroids), axis=1)

    def _compute_loss(self):
        """
        Compute the loss of the current clustering as the sum of the squared distances between the data points and the
        centroids, given the current cluster assignments
        :return: average loss
        """
        loss = 0
        for i in range(self.k):
            if len(self.clusters[i]) > 0 and len(self.centroids[i]) > 0:
                loss += np.sum([np.linalg.norm(self.clusters[i] - self.centroids[i]) ** 2])
        return loss / self.data.shape[0]

    def _NMI(self):
        """
        Calculate the NMI considering the true labels and the predicted labels
        :return: NMI
        """
        return normalized_mutual_info_score(self.true_labels, self.labels)
