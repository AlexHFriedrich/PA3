import os
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

from LloydsAlgorithm import LloydsAlgorithm
from LSHLloyd import LloydsAlgorithmLSH
from coresets import Coresets


def write_results_to_file(filename, nmi, number_dist_calc, runtime, num_iter):
    """
    Write the training results to corresponding file
    :param filename: location of the file
    :param nmi: achieved NMI
    :param number_dist_calc: number of distance calculations
    :param runtime: runtime in seconds
    :param num_iter: number of iterations for convergence
    :return:
    """
    with open(filename, 'w') as f:
        f.write('Average NMI: {}\n'.format(np.mean(nmi)))
        f.write('Average number of distance calculations: {}\n'.format(np.mean(number_dist_calc)))
        f.write('Average runtime: {}\n'.format(np.mean(runtime)))
        f.write('Average number of iterations: {}\n'.format(np.mean(num_iter)))


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '4'

    # set the seed for reproducibility
    np.random.seed(42)

    # maximum number of iterations the algorithms are trained for
    n_iter = 500

    # Load data and preprocess, dropping columns 0, 1, 2
    data = pd.read_csv('bio_train.csv', header=None)
    true_labels = data[0].tolist()
    data = data.drop(columns=[0, 1, 2]).to_numpy()
    num_clusters = len(set(true_labels))

    # scaling the data to improve performance
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    stats_for_comparison = {"Baseline": {"NMI": 0, "runtime": 0},
                            "LSH": {"NMI": 0, "runtime": 0},
                            "Coreset_100": {"NMI": 0, "runtime": 0},
                            "Coreset_1000": {"NMI": 0, "runtime": 0},
                            "Coreset_10000": {"NMI": 0, "runtime": 0}}

    # Lloyds Algorithm
    print("\n---------------LLOYD'S ALGORITHM---------------")

    # container to store results
    NMI = []
    losses = []
    num_distance_calculations = []
    runtimes = []
    num_iterations = []

    # run parameters for Lloyds Algorithm, setting random_init to false will use the first k data points as centroids
    random_init = True
    num_rep = 5 if random_init else 1

    for _ in range(num_rep):
        lloyds = LloydsAlgorithm(num_clusters, data, true_labels, max_iter=n_iter, random_init=random_init)
        lloyds.fit()
        NMI.append(lloyds.NMI)
        losses.append(lloyds.losses)
        num_distance_calculations.append(lloyds.num_distance_calculations)
        runtimes.append(lloyds.time)
        num_iterations.append(lloyds.n_iter_)

    print('Average NMI: {}'.format(np.mean(NMI)))
    print('Average number of distance calculations: {}'.format(np.mean(num_distance_calculations)))
    print('Average runtime: {}'.format(np.mean(runtimes)))
    print('Average number of iterations: {}'.format(np.mean(num_iterations)))

    stats_for_comparison["Baseline"]["NMI"] = np.mean(NMI)
    stats_for_comparison["Baseline"]["runtime"] = np.mean(runtimes)

    write_results_to_file('results/lloyds_algorithm_results.txt', np.mean(NMI), np.mean(num_distance_calculations),
                          np.mean(runtimes), np.mean(num_iterations))

    # plot the convergence of the algorithm
    for loss in losses:
        plt.plot(loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Convergence of Lloyd\'s Algorithm')
    plt.savefig('results/lloyds_algorithm_convergence.png')
    plt.close()

    # Lloyds Algorithm with LSH
    print("\n---------------LLOYD'S ALGORITHM WITH LSH---------------")
    # Container to store results
    NMI_lsh = []
    losses_lsh = []
    num_distance_calculations_lsh = []
    runtimes_lsh = []
    num_iterations_lsh = []

    for _ in range(5):
        lloyds_lsh = LloydsAlgorithmLSH(num_clusters, data, true_labels, num_hash_tables=5,
                                        num_hashes_per_table=5, bucket_size=4.0, max_iter=n_iter,
                                        debug=False)
        lloyds_lsh.fit()
        NMI_lsh.append(lloyds_lsh.NMI)
        losses_lsh.append(lloyds_lsh.losses)
        num_distance_calculations_lsh.append(lloyds_lsh.num_distance_calculations)
        runtimes_lsh.append(lloyds_lsh.time)
        num_iterations_lsh.append(lloyds_lsh.n_iter_)

    print('Average NMI: {}'.format(np.mean(NMI_lsh)))
    print('Average number of distance calculations: {}'.format(np.mean(num_distance_calculations_lsh)))
    print('Average runtime: {}'.format(np.mean(runtimes_lsh)))
    print('Average number of iterations: {}'.format(np.mean(num_iterations_lsh)))

    filename = 'results/lloyds_algorithm_lsh_results.txt'
    write_results_to_file(filename, np.mean(NMI_lsh),
                          np.mean(num_distance_calculations_lsh), np.mean(runtimes_lsh),
                          np.mean(num_iterations_lsh))

    # plot the convergence of the algorithm
    for loss_lsh in losses_lsh:
        plt.plot(loss_lsh)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Convergence of Lloyd\'s Algorithm with LSH')
    plt.savefig('results/lloyds_algorithm_lsh_convergence.png')
    plt.close()

    stats_for_comparison["LSH"]["NMI"] = np.mean(NMI_lsh)
    stats_for_comparison["LSH"]["runtime"] = np.mean(runtimes_lsh)

    # Coreset
    print("---------------CORESET---------------")

    coreset_sizes = [100, 1000, 10000]

    # container to store results
    NMI_scores = {size: [] for size in coreset_sizes}
    NMI_average_scores = {size: [] for size in coreset_sizes}
    num_it = {size: [] for size in coreset_sizes}
    num_distance_calculations = {size: [] for size in coreset_sizes}
    times = {size: [] for size in coreset_sizes}

    for size in coreset_sizes:
        for k in range(10):
            start = time.time()
            coreset = Coresets(min(num_clusters, int(size * 0.9)), data, true_labels, size)
            coreset.fit()
            pred = coreset.predict(data)

            times[size].append(time.time() - start)
            num_it[size].append(coreset.kmeans.n_iter_)
            NMI_scores[size].append(normalized_mutual_info_score(true_labels, pred, average_method='arithmetic'))
            num_distance_calculations[size].append(coreset.num_distance_calculations)

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

    # create a box plot for the NMI scores of different sizes
    plt.boxplot([NMI_scores[size] for size in coreset_sizes], labels=[str(size) for size in coreset_sizes])
    plt.xlabel('Coreset size')
    plt.ylabel('NMI')
    plt.title('NMI scores for different coreset sizes')
    plt.savefig('results/coreset_NMI_scores.png')
    plt.close()

    for size in coreset_sizes:
        file_name = 'results/coreset_{}_results.txt'.format(size)
        write_results_to_file(file_name, NMI_scores[size], num_distance_calculations[size], times[size], num_it[size])

    # NMI and runtime plots for different implementations
    # one color per implementation, axes of the scatterplot are average NMI and runtime

    plt.scatter([stats_for_comparison["Baseline"]["NMI"]], [stats_for_comparison["Baseline"]["runtime"]],
                label="Baseline")
    plt.scatter([stats_for_comparison["LSH"]["NMI"]], [stats_for_comparison["LSH"]["runtime"]], label="LSH")
    plt.scatter([stats_for_comparison["Coreset_100"]["NMI"]], [stats_for_comparison["Coreset_100"]["runtime"]],
                label="Coreset_100")
    plt.scatter([stats_for_comparison["Coreset_1000"]["NMI"]], [stats_for_comparison["Coreset_1000"]["runtime"]],
                label="Coreset_1000")
    plt.scatter([stats_for_comparison["Coreset_10000"]["NMI"]], [stats_for_comparison["Coreset_10000"]["runtime"]],
                label="Coreset_10000")
    plt.xlabel("NMI")
    plt.ylabel("Runtime")
    plt.legend()
    plt.savefig("results/NMI_vs_runtime.png")
