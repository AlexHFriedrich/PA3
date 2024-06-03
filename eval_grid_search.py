def read_grid_search_results_from_file(filename):
    results = {}
    with open(filename) as f:
        for line in f:
            key, value = line.strip().split(": ")
            results[key] = value
    return results


if __name__ == "__main__":
    results = read_grid_search_results_from_file("results_hyperparameter_grid_search_lsh.txt")
    for key, value in results.items():
        print(f"{key}: {value}")
    print("Number of configurations: ", len(results))

    # take the results and exclude those with an NMI < 0.1, as well as those, whose third value is less than 10000 or more than 100000
    filtered_results = {key: value for key, value in results.items() if
                        float(value.split(", ")[0][1:-1]) > 0.1 and 10000 < float(
                            value.split(", ")[2].strip(']')) < 100000}

    # return the key value pair that maximizes the NMI
    best_configuration = max(filtered_results.items(), key=lambda x: float(x[1].split(", ")[0][1:-1]))

    print("Best configuration: ", best_configuration)
