import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from LloydsAlgorithm import LloydsAlgorithm


class Coresets:
    def __init__(self, k, data, true_labels, coreset_size, max_iter=1000):
        self.kmeans = None
        self.k = k
        self.data = data
        self.true_labels = true_labels
        self.coreset_size = coreset_size
        self.num_distance_calculations = 0
        self.max_iter = max_iter
        self.coreset_idx, self.weights = self._build_coreset()

    def _build_coreset(self):
        mean = np.mean(self.data, axis=0)
        total_mean_distance = np.sum(
            pairwise_distances(self.data, mean.reshape(1, -1), metric='euclidean').flatten() ** 2)
        probs = self._calculate_probabilities(mean, total_mean_distance)
        coreset_idx = self._sample(probs)
        weights = self._calculate_weights(probs)
        return coreset_idx, weights

    def _calculate_probabilities(self, mean, total_mean_distance):
        return 1 / 2 * (1 / len(self.data) + np.linalg.norm(self.data - mean, axis=1) ** 2 / total_mean_distance)

    def _sample(self, probs):
        return np.random.choice(len(self.data), size=self.coreset_size, p=probs)

    def _calculate_weights(self, probs):
        return 1 / (self.coreset_size * probs)

    def fit(self):
        self.kmeans = KMeans(n_clusters=self.k, init='random', max_iter=self.max_iter)
        self.kmeans.fit(self.data[self.coreset_idx], sample_weight=self.weights[self.coreset_idx])
        self.num_distance_calculations += self.kmeans.n_iter_ * self.coreset_size * self.k

    def fit_Lloyds(self):
        self.kmeans = LloydsAlgorithm(self.k, self.data[self.coreset_idx],
                                      list(np.array(self.true_labels)[self.coreset_idx]), max_iter=self.max_iter)
        self.kmeans.fit()
        self.num_distance_calculations += self.kmeans.num_distance_calculations

    def predict(self, data):
        self.num_distance_calculations += data.shape[0] * self.k
        return self.kmeans.predict(data)
