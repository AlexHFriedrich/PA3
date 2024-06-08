The implementation of Lloyd's algorithm found in the `LloydsAlgorithm.py` file, follows the standard algorithm for KMeans.
Upon initialization of the algorithm, the initial centroids are either chosen to be the first k points in the data, or 
randomly chosen from the data points.
Everything else happens in the `fit` method, which iterates at most `max_iter` times, or until the centroids do not 
change anymore.
Each iteration starts by assigning each data point to a cluster and check for convergence, which is both handled within 
the `_assign_clusters` method.
If the algorithm has not converged, the centroids are then updated, handled by the `_update_centroids` method. Additionally current loss is computed and additional metrics stored. After convergence, or after the maximum number of iterations are reached NMI and the time spend training are computed and stored.

