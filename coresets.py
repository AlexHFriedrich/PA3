from KMeans import KMeans


class Coresets(KMeans):
    def __init__(self, k, data):
        super().__init__(k, data)

    def initialize_centroids(self):
        return None

    def update_centroids(self):
        return None

    def assign_clusters(self):
        return None

    def compute_loss(self):
        return None

    def convergence_plot(self):
        return None