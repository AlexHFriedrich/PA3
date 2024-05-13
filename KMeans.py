class KMeans:
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.iterations = 0
        self.centroids = self.initialize_centroids()
        self.losses = []
        self.NMI = 0

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