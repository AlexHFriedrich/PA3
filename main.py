import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from LloydsAlgorithm import LloydsAlgorithm
from LSHLloyd import LloydsAlgorithmLSH

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    data = pd.read_csv('bio_train.csv', header=None)
    true_labels = data[0].tolist()
    data = data.drop(columns=[0,1,2]).to_numpy()
    num_clusters = len(set(true_labels))
    print(data.shape)
    print(num_clusters)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    print("---------------SKLEARN---------------")
    kmeans_sklearn = KMeans(n_clusters=num_clusters, init=data.copy()[:num_clusters], max_iter=1000)
    pred_labels = kmeans_sklearn.fit_predict(data.copy())
    print('NMI: {}'.format(normalized_mutual_info_score(true_labels, pred_labels)))

    # Lloyds Algorithm
    print("---------------LLOYD ALGORITHM---------------")
    NMI = []
    losses = []
    num_distance_calculations = []
    runtimes = []
    num_iterations = []
    for _ in range(1):
        lloyds = LloydsAlgorithm(num_clusters, data.copy(), true_labels, max_iter=1000)
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
    plt.close()
    #plt.show()


    # # Lloyds Algorithm with LSH
    # print("---------------LLOYD ALGORITHM WITH LSH---------------")
    # NMI_lsh = []
    # losses_lsh = []
    # num_distance_calculations_lsh = []
    # runtimes_lsh = []
    # num_iterations_lsh = []
    # for _ in range(1):
    #     lloyds_lsh = LloydsAlgorithmLSH(num_clusters, data.copy(), true_labels, num_hash_tables=3, num_hashes_per_table=3, bucket_size=4.0, max_iter=1000, debug=True)
    #     lloyds_lsh.fit()
    #     NMI_lsh.append(lloyds_lsh.NMI)
    #     losses_lsh.append(lloyds_lsh.losses)
    #     num_distance_calculations_lsh.append(lloyds_lsh.num_distance_calculations)
    #     runtimes_lsh.append(lloyds_lsh.time)
    #     num_iterations_lsh.append(lloyds_lsh.iterations)
    # print('Average NMI: {}'.format(np.mean(NMI_lsh)))
    # print('Average number of distance calculations: {}'.format(np.mean(num_distance_calculations_lsh)))
    # print('Average runtime: {}'.format(np.mean(runtimes_lsh)))
    # print('Average number of iterations: {}'.format(np.mean(num_iterations_lsh)))

    # with open('lloyds_algorithm_lsh_results.txt', 'w') as f:
    #     f.write('Average NMI: {}\n'.format(np.mean(NMI_lsh)))
    #     f.write('Average number of distance calculations: {}\n'.format(np.mean(num_distance_calculations_lsh)))
    #     f.write('Average runtime: {}\n'.format(np.mean(runtimes_lsh)))
    #     f.write('Average number of iterations: {}\n'.format(np.mean(num_iterations_lsh)))

    # for loss_lsh in losses_lsh:
    #     plt.plot(loss_lsh)
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Loss')
    #     plt.title('Convergence of Lloyd\'s Algorithm with LSH')
    # plt.savefig('lloyds_algorithm_lsh_convergence.png')
    # plt.close()
    # #plt.show()
