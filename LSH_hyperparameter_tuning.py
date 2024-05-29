import pandas as pd
from sklearn.preprocessing import StandardScaler

from LSHLloyd import LloydsAlgorithmLSH

if __name__ == "__main__":
    n_iter = 50

    # Load data and preprocess
    data = pd.read_csv('bio_train.csv', header=None)
    true_labels = data[0].tolist()
    data = data.drop(columns=[0, 1, 2]).to_numpy()
    num_clusters = len(set(true_labels))

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    grid = {"num_hash_tables": [2, 5, 10], "num_hashes_per_table": [2, 5, 10], "bucket_size": [1.0, 2.0, 4.0]}

    optimal_params = {"num_hash_tables": 2, "num_hashes_per_table": 2, "bucket_size": 1.0, "NMI": 0}

    for num_hash_tables in grid["num_hash_tables"]:
        for num_hashes_per_table in grid["num_hashes_per_table"]:
            for bucket_size in grid["bucket_size"]:
                NMI_lsh = []
                losses_lsh = []
                num_distance_calculations_lsh = []
                runtimes_lsh = []
                num_iterations_lsh = []

                for _ in range(5):
                    lloyds_lsh = LloydsAlgorithmLSH(num_clusters, data.copy(), true_labels,
                                                    num_hash_tables=num_hash_tables,
                                                    num_hashes_per_table=num_hashes_per_table, bucket_size=bucket_size,
                                                    max_iter=n_iter,
                                                    debug=False)
                    lloyds_lsh.fit()
                    NMI_lsh.append(lloyds_lsh.NMI)
                    losses_lsh.append(lloyds_lsh.losses)
                    num_distance_calculations_lsh.append(lloyds_lsh.num_distance_calculations)
                    runtimes_lsh.append(lloyds_lsh.time)
                    num_iterations_lsh.append(lloyds_lsh.iterations)

                NMI = sum(NMI_lsh) / len(NMI_lsh)

                if NMI > optimal_params["NMI"]:
                    optimal_params["num_hash_tables"] = num_hash_tables
                    optimal_params["num_hashes_per_table"] = num_hashes_per_table
                    optimal_params["bucket_size"] = bucket_size
                    optimal_params["NMI"] = NMI

    print(optimal_params)