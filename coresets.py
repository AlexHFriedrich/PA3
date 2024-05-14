from KMeans import KMeans


class Coresets(KMeans):
    def __init__(self, k, data, true_labels):
        super().__init__(k, data, true_labels)

    def fit(self):
        pass

    def initialize_centroids(self):
        return None

    def update_centroids(self):
        return None

    def assign_clusters(self):
        return None
