import gc
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import StandardScaler

from LSHLloyd import LloydsAlgorithmLSH

if __name__ == "__main__":
    n_iter = 10

    # Load data and preprocess
    data = pd.read_csv('bio_train.csv', header=None)
    true_labels = data[0].tolist()
    data = data.drop(columns=[0, 1, 2]).to_numpy()
    num_clusters = len(set(true_labels))

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    grid = {"num_hash_tables": [2, 3, 4, 5, 6, 7, 8, 9, 10], "num_hashes_per_table": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "bucket_size": [0.5, 1.0, 2.0, 4.0, 8.0]}

    results = defaultdict(list)

    for num_hash_tables in grid["num_hash_tables"]:
        for num_hashes_per_table in grid["num_hashes_per_table"]:
            if abs(num_hash_tables - num_hashes_per_table) < 3:
                for bucket_size in grid["bucket_size"]:
                    NMI_lsh = []
                    runtimes_lsh = []
                    num_assigned_values = []

                    for _ in range(3):
                        lloyds_lsh = LloydsAlgorithmLSH(num_clusters, data.copy(), true_labels,
                                                        num_hash_tables=num_hash_tables,
                                                        num_hashes_per_table=num_hashes_per_table,
                                                        bucket_size=bucket_size,
                                                        max_iter=n_iter,
                                                        debug=False)
                        lloyds_lsh.fit()
                        NMI_lsh.append(lloyds_lsh.NMI)
                        runtimes_lsh.append(lloyds_lsh.time)
                        num_assigned_values.append(lloyds_lsh.num_assignments)

                    NMI = sum(NMI_lsh) / len(NMI_lsh)
                    config_str = f"nht={num_hash_tables}, nhpt={num_hashes_per_table}, bs={bucket_size}"
                    results[config_str] = [NMI, sum(runtimes_lsh) / len(runtimes_lsh),
                                           sum(num_assigned_values) / len(num_assigned_values)]
                    gc.collect()

    for key, value in results.items():
        print(f"{key}: {value}")

    with open('results_hyperparameter_grid_search_lsh.txt', 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
