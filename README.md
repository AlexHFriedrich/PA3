# PA 3 - Alexander Friedrich - Christoph Wolf

In this assignment we implement different strategies to update cluster centroids for the KMeans algorithm. Evaluation and
code to produce the results can be found in the main.py file, while each implementation is found in the corresponding
python file. 

- `kmeans.py` contains a parent class, which is inherited by the different strategies. It contains the main logic of the
  KMeans algorithm.
- `LloydsAlgortihm.py` contains the implementation of the Lloyd's algorithm for centroid updates
- `LSH.py` contains an approach using Locality Sensitive Hashing for centroid updates
- `coresets.py` contains an approach using coresets to find an optimal clustering