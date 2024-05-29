import random

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, normalized_mutual_info_score
import os

from sklearn.preprocessing import StandardScaler


class Coresets:
    def __init__(self, k, data, true_labels, coreset_size):
        self.k = k
        self.data = data
        self.true_labels = true_labels
        self.coreset_size = coreset_size

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
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(self.data[self.coreset_idx], sample_weight=self.weights[self.coreset_idx])
        return kmeans


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'

    data = pd.read_csv('bio_train.csv', header=None)
    true_labels = data[0].tolist()
    data = data.drop(columns=[0, 1, 2]).to_numpy()
    num_clusters = len(set(true_labels))

    data = StandardScaler().fit_transform(data)

    coreset_sizes = [100, 1000, 10000]
    NMI_scores = {size: [] for size in coreset_sizes}
    NMI_average_scores = {size: [] for size in coreset_sizes}
    num_it = {size: [] for size in coreset_sizes}
    for size in coreset_sizes:
        for k in range(10):
            coreset = Coresets(min(num_clusters, int(size * 0.9)), data, true_labels, size)
            predictor = coreset.fit()
            pred = predictor.predict(data)
            num_it[size].append(predictor.n_iter_)
            NMI_scores[size].append(normalized_mutual_info_score(true_labels, pred, average_method='arithmetic'))
        NMI_average_scores[size] = np.mean(NMI_scores[size])
        num_it[size] = np.mean(num_it[size])

    print(NMI_average_scores)
    print(num_it)

    # check the variance of the NMI scores for each coreset size
    for size in coreset_sizes:
        print(np.var(NMI_scores[size]))
