import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances, normalized_mutual_info_score
from tqdm import trange, tqdm


class KMeans:
    def __init__(self, k, data, true_labels, max_iter):
        self.k = k
        self.data = data
        self.iterations = 0
        self.losses = []
        self.NMI = 0
        self.true_labels = self._update_true_labels(true_labels)
        self.clusters = dict()
        self.centroids = self.initialize_centroids()
        self.labels = np.zeros(self.data.shape[0])
        self.max_iter = max_iter

    def fit(self):
        raise NotImplementedError

    def initialize_centroids(self):
        raise NotImplementedError

    def update_centroids(self):
        raise NotImplementedError

    def assign_clusters(self):
        raise NotImplementedError

    def compute_loss(self):
        loss = 0
        for i in range(self.k):
            loss += np.sum(pairwise_distances(self.clusters[i], [self.centroids[i]]))
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
