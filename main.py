import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from LloydsAlgorithm import LloydsAlgorithm

if __name__ == '__main__':
    data = pd.read_csv('bio_train.csv', header=None)
    true_labels = data[0].tolist()
    data = data.drop(columns=[0, 1, 2]).to_numpy()
    num_clusters = len(set(true_labels))

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    kmeans_sklearn = KMeans(n_clusters=num_clusters, init='random', max_iter=1000, n_init=5)
    pred_labels = kmeans_sklearn.fit_predict(data)
    print('NMI: {}'.format(normalized_mutual_info_score(true_labels, pred_labels)))

    # Lloyds Algorithm
    NMI = []
    losses = []
    num_distance_calculations = []
    runtimes = []
    num_iterations = []
    for _ in range(5):
        lloyds = LloydsAlgorithm(num_clusters, data, true_labels)
        lloyds.fit()
        NMI.append(lloyds.NMI)
        losses.append(lloyds.losses)
        num_distance_calculations.append(lloyds.num_distance_calculations)
        runtimes.append(lloyds.time)
        num_iterations.append(lloyds.iterations)
    print('Average NMI: {}'.format(np.mean(NMI)))
    print('Average number of distance calculations: {}'.format(np.mean(num_distance_calculations)))
    print('Average runtime: {}'.format(np.mean(runtimes)))
    print('Average number of iterations: {}'.format(np.mean(num_iterations)))

    with open('lloyds_algorithm_results.txt', 'w') as f:
        f.write('Average NMI: {}\n'.format(np.mean(NMI)))
        f.write('Average number of distance calculations: {}\n'.format(np.mean(num_distance_calculations)))
        f.write('Average runtime: {}\n'.format(np.mean(runtimes)))
        f.write('Average number of iterations: {}\n'.format(np.mean(num_iterations)))

    for loss in losses:
        plt.plot(loss)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Convergence of Lloyd\'s Algorithm')
    plt.savefig('lloyds_algorithm_convergence.png')
    plt.show()
