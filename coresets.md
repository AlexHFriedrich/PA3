The implementation of coresets for Kmeans, found in the `coresets.py` file, is based on the paper provided in the
assignment.

Upon initialization the coreset and the corresponding weights are computed. This is handled within the `_build_coreset`
method that samples points from the data set, based on algorithm 1 from the paper. Everything else is then handled by
the
sklearn implementation of kmeans, that fits a clustering model to the coreset, which is then used to predict all points
in the data.

# Comment for the variance of the accuracy
The variance of the accuracy is computed by running the algorithm multiple times and computing the variance of the 
recorded NMI scores over all runs. This is then represented using a boxplot. In this we can see that the mean of the NMI 
scores increases with the size of the coreset for our choice of sizes, while the variance decreases. This is to be 
expected, as a larger coreset should be more representative of the data, and thus yield smaller variance through 
sampling.
