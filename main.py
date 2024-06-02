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
    n_iter = 50

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

    stats_for_comparison = {"Baseline": {"NMI": 0, "runtime": 0},
                            "LSH": {"NMI": 0, "runtime": 0},
                            "Coreset_100": {"NMI": 0, "runtime": 0},
                            "Coreset_1000": {"NMI": 0, "runtime": 0},
                            "Coreset_10000": {"NMI": 0, "runtime": 0}}


    # Lloyds Algorithm
    print("---------------LLOYD ALGORITHM---------------")
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

    stats_for_comparison["Baseline"]["NMI"] = np.mean(NMI)
    stats_for_comparison["Baseline"]["runtime"] = np.mean(runtimes)

    write_results_to_file('lloyds_algorithm_results.txt', np.mean(NMI), np.mean(num_distance_calculations),
                          np.mean(runtimes), np.mean(num_iterations))

    for loss in losses:
        plt.plot(loss)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Convergence of Lloyd\'s Algorithm')
    plt.savefig('lloyds_algorithm_convergence.png')
    plt.close()

    # Lloyds Algorithm with LSH
    print("---------------LLOYD ALGORITHM WITH LSH---------------")
    configs = [(1, 2)]
    for config in configs:

        NMI_lsh = []
        losses_lsh = []
        num_distance_calculations_lsh = []
        runtimes_lsh = []
        num_iterations_lsh = []
        for _ in range(5):
            lloyds_lsh = LloydsAlgorithmLSH(num_clusters, data.copy(), true_labels, num_hash_tables=config[0],
                                            num_hashes_per_table=config[1], bucket_size=1.0, max_iter=n_iter,
                                            debug=False)
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

        filename = 'lloyds_algorithm_lsh_results_{}_{}.txt'.format(config[0], config[1])
        write_results_to_file(filename, np.mean(NMI_lsh),
                              np.mean(num_distance_calculations_lsh), np.mean(runtimes_lsh),
                              np.mean(num_iterations_lsh))

        for loss_lsh in losses_lsh:
            plt.plot(loss_lsh)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Convergence of Lloyd\'s Algorithm with LSH')
        plt.savefig('lloyds_algorithm_lsh_convergence.png')
        plt.close()

        stats_for_comparison["LSH"]["NMI"] = np.mean(NMI_lsh)
        stats_for_comparison["LSH"]["runtime"] = np.mean(runtimes_lsh)

    # Coreset
    print("---------------CORESET---------------")

    coreset_sizes = [100, 1000, 10000]
    NMI_scores = {size: [] for size in coreset_sizes}
    NMI_average_scores = {size: [] for size in coreset_sizes}
    num_it = {size: [] for size in coreset_sizes}
    num_distance_calculations = {size: [] for size in coreset_sizes}
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
            num_distance_calculations[size].append(
                predictor.n_iter_ * size * num_clusters + data.shape[0] * num_clusters)
        NMI_average_scores[size] = np.mean(NMI_scores[size])
        num_it[size] = np.mean(num_it[size])
        times[size] = np.mean(times[size])
        num_distance_calculations[size] = np.mean(num_distance_calculations[size])

        stats_for_comparison["Coreset_{}".format(size)]["NMI"] = np.mean(NMI_scores[size])
        stats_for_comparison["Coreset_{}".format(size)]["runtime"] = np.mean(times[size])

    for size in coreset_sizes:
        print('Coreset size: {}'.format(size))
        print('Average NMI: {}'.format(NMI_average_scores[size]))
        print('Average number of iterations: {}'.format(num_it[size]))
        print('Average number of distance calculations: {}'.format(num_distance_calculations[size]))
        print('Average runtime: {}'.format(times[size]))
    '''# check the variance of the NMI scores for each coreset size
    for size in coreset_sizes:
        print(np.var(NMI_scores[size]))'''

    for size in coreset_sizes:
        file_name = 'coreset_{}_results.txt'.format(size)
        write_results_to_file(file_name, NMI_scores[size], num_distance_calculations[size], times[size], num_it[size])


    # NMI and runtime plots for different implementations
    # one color per implementation, axes of the scatterplot are average NMI and runtime

    plt.scatter([stats_for_comparison["Baseline"]["NMI"]], [stats_for_comparison["Baseline"]["runtime"]], label="Baseline")
    plt.scatter([stats_for_comparison["LSH"]["NMI"]], [stats_for_comparison["LSH"]["runtime"]], label="LSH")
    plt.scatter([stats_for_comparison["Coreset_100"]["NMI"]], [stats_for_comparison["Coreset_100"]["runtime"]], label="Coreset_100")
    plt.scatter([stats_for_comparison["Coreset_1000"]["NMI"]], [stats_for_comparison["Coreset_1000"]["runtime"]], label="Coreset_1000")
    plt.scatter([stats_for_comparison["Coreset_10000"]["NMI"]], [stats_for_comparison["Coreset_10000"]["runtime"]], label="Coreset_10000")
    plt.xlabel("NMI")
    plt.ylabel("Runtime")
    plt.legend()
    plt.savefig("NMI_vs_runtime.png")
