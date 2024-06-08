The LloydsAlgorithmLSH class is a version of the normal k-means clustering algorithm that uses Locality-Sensitive Hashing (LSH). This class extends our normal Lloyd's algorithm by integrating LSH to reduce the number of distance calculations between points and centroids.

# Implementation

The implementation of LSH for Kmeans, found in the LSHLloyd.py file, is based on the description provided in the pdf on Moodle:

Upon initialization, the class sets up several parameters, generates the required hash functions and hash tables, and hashes the data points into these tables, grouping similar points together in hash buckets.

The fit method in our class iteratively updates the centroids and reassigns points to clusters until convergence or the maximum number of iterations is reached. This process includes hashing the centroids, assigning points to clusters based on hash buckets, handling unassigned points by direct distance calculation, and updating centroids based on the new assignments.

We created an organized hash functions like follows:

- A vector **a** generated from a normal distribution (np.random.normal), which matches the dimensionality of the data.
- A scalar **b** drawn from a uniform distribution (np.random.uniform), within the range of the bucket size.

Each hash function uses the formula hash_value=⌊(point⋅a+b)/bucket_size⌋, where the dot product between the data point and the vector a is computed, shifted by b, and divided by the bucket size before applying the floor function.

Multiple hash functions are grouped into hash tables. The number of hash tables (num_hash_tables) and the number of hash functions per table (num_hashes_per_table) are configurable parameters.

When a data point is hashed, it is processed by each hash function in a table, resulting in a tuple of hash values (one per hash function). This tuple serves as the key for the hash bucket.If a datapoint hashes to the same bucket as one of the centroids in at least one hashing table, this is considered a match and the data point is assigned to that centroid. If it matches with multiple centroids, one of them is picked randomly.

# Accuracy, number of distance calculations and runtime

The average Normalized Mutual Information (NMI) for the selected hyperparameters (num_hash_tables=3, num_hashes_per_table=4, bucket_size=4.0) accounts for a relatively lower level of about 0.1285. When selecting the hyperparameters, we encountered a trade-off between runtime and accuracy. The more points we could assign during the hashing part, the fewer distance calculations were necessary, which reduced the runtime. However, this often came at the cost of lower clustering accuracy. During our grid search, we aimed to find a good balance between these competing factors, settling on hyperparameters that provided a reasonable compromise.
Our algorithm did not converge before reaching the maximum number of iterations (500). This was because, in each iteration, some points changed their buckets during the hashing assignment, preventing the clusters from stabilizing completely. Despite this, the average total number of distance calculations across all iterations was significantly reduced, amounting to 3,823,708,098.6.
