import os
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from LloydsAlgorithm import LloydsAlgorithm
from LSHLloyd import LloydsAlgorithmLSH
from coresets import Coresets


def write_results_to_file(filename, nmi, number_dist_calc, runtime, num_iter):
    with open(filename, 'w') as f:
        f.write('Average NMI: {}\n'.format(np.mean(nmi)))
        f.write('Average number of distance calculations: {}\n'.format(np.mean(number_dist_calc)))
        f.write('Average runtime: {}\n'.format(np.mean(runtime)))
        f.write('Average number of iterations: {}\n'.format(np.mean(num_iter)))


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    n_iter = 100

    # Load data and preprocess
    data = pd.read_csv('bio_train.csv', header=None)
    true_labels = data[0].tolist()
    data = data.drop(columns=[0, 1, 2]).to_numpy()
    num_clusters = len(set(true_labels))

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Baseline using sklearn
    print("---------------SKLEARN---------------")
    kmeans_sklearn = KMeans(n_clusters=num_clusters, init=data.copy()[:num_clusters], max_iter=n_iter)
    pred_labels = kmeans_sklearn.fit_predict(data.copy())
    print('NMI: {}'.format(normalized_mutual_info_score(true_labels, pred_labels)))

    # Lloyds Algorithm
    """print("---------------LLOYD ALGORITHM---------------")
    NMI = []
    losses = []
    num_distance_calculations = []
    runtimes = []
    num_iterations = []
    for _ in range(1):
        lloyds = LloydsAlgorithm(num_clusters, data.copy(), true_labels, max_iter=n_iter)
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

    write_results_to_file('lloyds_algorithm_results.txt', NMI, num_distance_calculations, runtimes, num_iterations)

    for loss in losses:
        plt.plot(loss)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Convergence of Lloyd\'s Algorithm')
    plt.savefig('lloyds_algorithm_convergence.png')
    plt.close()"""

    # Lloyds Algorithm with LSH
    print("---------------LLOYD ALGORITHM WITH LSH---------------")
    configs = [(1, 2), (2, 1)]
    for config in configs:

        NMI_lsh = []
        losses_lsh = []
        num_distance_calculations_lsh = []
        runtimes_lsh = []
        num_iterations_lsh = []
        for _ in range(3):
            lloyds_lsh = LloydsAlgorithmLSH(num_clusters, data.copy(), true_labels, num_hash_tables=config[0],
                                            num_hashes_per_table=config[1], bucket_size=1.0, max_iter=n_iter, debug=False)
            lloyds_lsh.fit()
            NMI_lsh.append(lloyds_lsh.NMI)
            losses_lsh.append(lloyds_lsh.losses)
            num_distance_calculations_lsh.append(lloyds_lsh.num_distance_calculations)
            runtimes_lsh.append(lloyds_lsh.time)
            num_iterations_lsh.append(lloyds_lsh.iterations)
        print('Average NMI: {}'.format(np.mean(NMI_lsh)))
        print('Average number of distance calculations: {}'.format(np.mean(num_distance_calculations_lsh)))
        print('Average runtime: {}'.format(np.mean(runtimes_lsh)))
        print('Average number of iterations: {}'.format(np.mean(num_iterations_lsh)))

        write_results_to_file('lloyds_algorithm_lsh_results.txt', NMI_lsh, num_distance_calculations_lsh, runtimes_lsh,
                              num_iterations_lsh)

        for loss_lsh in losses_lsh:
            plt.plot(loss_lsh)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Convergence of Lloyd\'s Algorithm with LSH')
        plt.savefig('lloyds_algorithm_lsh_convergence.png')
        plt.close()

    # Coreset
    print("---------------CORESET---------------")

    coreset_sizes = [100, 1000, 10000]
    NMI_scores = {size: [] for size in coreset_sizes}
    NMI_average_scores = {size: [] for size in coreset_sizes}
    num_it = {size: [] for size in coreset_sizes}
    times = {size: [] for size in coreset_sizes}
    for size in coreset_sizes:
        for k in range(10):
            start = time.time()
            coreset = Coresets(min(num_clusters, int(size * 0.9)), data, true_labels, size)
            predictor = coreset.fit()
            pred = predictor.predict(data)
            times[size].append(time.time() - start)
            num_it[size].append(predictor.n_iter_)
            NMI_scores[size].append(normalized_mutual_info_score(true_labels, pred, average_method='arithmetic'))
        NMI_average_scores[size] = np.mean(NMI_scores[size])
        num_it[size] = np.mean(num_it[size])
        times[size] = np.mean(times[size])

    for size in coreset_sizes:
        print('Coreset size: {}'.format(size))
        print('Average NMI: {}'.format(NMI_average_scores[size]))
        print('Average number of iterations: {}'.format(num_it[size]))
        print('Average runtime: {}'.format(times[size]))
    '''# check the variance of the NMI scores for each coreset size
    for size in coreset_sizes:
        print(np.var(NMI_scores[size]))'''

    for size in coreset_sizes:
        file_name = 'coreset_{}_results.txt'.format(size)
        write_results_to_file(file_name, NMI_scores[size], num_it[size], times[size], num_it[size])
