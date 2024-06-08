# PA 3 - Alexander Friedrich - Christoph Wolf

In this assignment we implement different strategies to solve the Kmeans-problem. Evaluation and
code to produce the results can be found in the main.py file, while each implementation is found in the corresponding
python file. 

- `LloydsAlgortihm.py` contains the implementation of the Lloyd's algorithm to perform KMeans
- `LSH.py` contains an approach using Locality Sensitive Hashing for KMeans
- `coresets.py` contains an approach using coresets to solve the KMeans problem

Testing was done using Python 3.12 and an environment with the following packages installed (most recent versions 
suffice) is necessary:
- pandas 
- numpy 
- matplotlib
- sklearn
- tqdm

To reproduce results run the main.py file. The results are saved in the `results` folder, in the corresponding txt-files 
and plots. Due to longer runtimes we would recommend to reduce the number of iterations in the main.py file, to e.g. 50 
iterations, to check that the code runs.
