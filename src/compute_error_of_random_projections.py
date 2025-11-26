import numpy as np
import random
import cProfile
import pstats
import io
import time
import struct
from scipy.stats import norm
import os
import gzip
import json
import shutil
import tempfile
import zipfile
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
import numpy as np

def random_projection(data, dimension):
    final_vector = np.zeros(dimension, dtype=np.float32)

    for element in data:
        # Process 32 dimensions at a time
        for d_start in range(0, dimension, 32):
            # Hash the (element, d_start) tuple
            h = hash((element, d_start))
            # Use 32 bits of the hash to determine signs
            for offset in range(min(32, dimension - d_start)):
                sign = 1 if ((h >> offset) & 1) == 0 else -1
                final_vector[d_start + offset] += sign
    final_vector /= np.sqrt(dimension)
    return final_vector

def get_me_a_random_projection_like_vector(dimension, number_of_elements):

    vec = np.random.binomial(number_of_elements, 0.5, dimension)
    vec = 2 * vec - number_of_elements
    vec = vec.astype(np.float32)
    vec /= np.sqrt(dimension)
    return vec

def plot_error_random_proj():
    # n_elements_list = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    # n_sets = 5000
    # dimension = 512
    # relative_errors = []

    # for n_elements in n_elements_list:
    #     projections = [get_me_a_random_projection_like_vector(dimension, n_elements) for _ in range(n_sets)]
    #     dot_products = []
    #     for i in range(2500):
    #         dot = np.dot(projections[2*i], projections[2*i+1])
    #         dot_products.append(dot)
    #     dot_products_sorted = sorted(dot_products)
    #     max_error = (dot_products_sorted[-10] - dot_products_sorted[10]) / 2
    #     relative_error = max_error / n_elements
    #     relative_errors.append(relative_error)
    #     print(f"n_elements={n_elements}, Max error: {max_error}, Relative error: {relative_error}")

    # plt.figure()
    # plt.plot(n_elements_list, relative_errors, marker='o')
    # plt.xscale('log')
    # plt.ylim([0,0.2])
    # plt.xlabel('|A u B| - |A n B|')
    # plt.ylabel('Error/(|A u B| - |A n B|)')
    # plt.title(f'Error/(|A u B| - |A n B|) vs |A u B| - |A n B|\nd={dimension}')
    # plt.grid(True)
    # plt.show()

    n_elements = 2000
    n_sets = 5000
    dimension_list = [256, 512, 1024, 2048, 4096, 8192, 16384]
    relative_errors = []

    for dimension in dimension_list:
        projections = [get_me_a_random_projection_like_vector(dimension, n_elements) for _ in range(n_sets)]
        dot_products = []
        for i in range(n_sets // 2):
            dot = np.dot(projections[2*i], projections[2*i+1])
            dot_products.append(dot)
        dot_products_sorted = sorted(dot_products)
        max_error = (dot_products_sorted[-10] - dot_products_sorted[10]) / 2
        relative_error = max_error / n_elements
        relative_errors.append(relative_error)
        print(f"dimension={dimension}, Max error: {max_error}, Relative error: {relative_error}")

    plt.figure()
    plt.plot(dimension_list, relative_errors, marker='o')
    plt.ylim([0,0.2])
    plt.xlabel('d (dimension)')
    plt.ylabel('error parameter s')
    plt.title(f'Error parameter s vs d\nn_elements={n_elements}')
    plt.grid(True)
    plt.show()

def compare_exact_and_dashing():

    exact_matrix = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/exact_ava.bin"
    # Load the exact matrix
    with open(exact_matrix, "rb") as f:
        exact_data = np.frombuffer(f.read(), dtype=np.int32).reshape(35003, 35003)

    exact_distances = []
    approximate_distances = []

    # Dashing - Efficient matrix reordering
    print("Reading name mappings...")
    output_file = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/dashing/reordered_jaccard_matrix.bin"

    # # Reorganize the Dashing matrix, it is in a different order compared to the stored exact matrix
    # # Read the current dashing order
    # dashing_names = []
    # with open("/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/dashing/sigs_dashinged/db_name.ss.names.txt", "r") as f:
    #     for line in f:
    #         if not line.startswith('#'):
    #             name = line.split()[0]
    #             dashing_names.append(name)

    # # Read the desired order
    # desired_names = []
    # with open("/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/ava_distances/vector_norms.txt", "r") as f:
    #     for line in f:
    #         name = line.split()[0]
    #         desired_names.append(name)

    # # Create mapping from dashing index to desired index
    # dashing_to_desired = {}
    # for dashing_idx, name in enumerate(dashing_names):
    #     if name in desired_names:
    #         desired_idx = desired_names.index(name)
    #         dashing_to_desired[dashing_idx] = desired_idx

    # print(f"Found mappings for {len(dashing_to_desired)} entries")

    # # Read the dashing matrix and reorder it
    # approximate_jaccard_data = np.zeros((35003, 35003), dtype=np.float32)

    # print("Reading and reordering dashing matrix...")
    # with open("/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/dashing/distances.txt", "r") as f:
    #     for line in f:
    #         if line[0] != '#':
    #             ls = line.split()
    #             dashing_row = int(ls[0])
    #             distances = [float(i) if i != '-' else -1 for i in ls[1:]]
                
    #             if dashing_row in dashing_to_desired:
    #                 desired_row = dashing_to_desired[dashing_row]
                    
    #                 for dashing_col in range(dashing_row + 1, len(distances)):
    #                     if dashing_col in dashing_to_desired:
    #                         desired_col = dashing_to_desired[dashing_col]
    #                         jaccard_val = distances[dashing_col]
    #                         approximate_jaccard_data[desired_row, desired_col] = jaccard_val
    #                         approximate_jaccard_data[desired_col, desired_row] = jaccard_val
                
    #             if dashing_row % 100 == 0:
    #                 print(f"Processed row {dashing_row}")
    #                 # if dashing_row > 1000:
    #                 #     break

    # print("Matrix reordering complete")
    
    # # Save the reorganized matrix
    # print(f"Saving reorganized matrix to {output_file}...")
    # with open(output_file, "wb") as f:
    #     approximate_jaccard_data.astype(np.float32).tofile(f)
    # print("Matrix saved successfully")

    # Load the reorganized matrix
    print("Loading reorganized matrix...")
    with open(output_file, "rb") as f:
        approximate_jaccard_data = np.frombuffer(f.read(), dtype=np.float32).reshape(35003, 35003)
    print("Matrix loaded successfully")

    # Check consistency with exact data
    print("\nChecking consistency with exact data...")
    consistency_checks = 0
    for i in range(min(100, exact_data.shape[0])):  # Check first 100 rows
        for j in range(min(100, exact_data.shape[1])):  # Check first 100 columns
            if i != j and exact_data[i, j] != 0:  # Skip diagonal and zero entries
                exact_jaccard = exact_data[i, j] / (exact_data[i, i] + exact_data[j, j] - exact_data[i, j])
                approx_jaccard = approximate_jaccard_data[i, j]
                diff = abs(exact_jaccard - approx_jaccard)
                if consistency_checks < 10:  # Print first 10 comparisons
                    print(f"Row {i}, Col {j}: Exact={exact_jaccard:.6f}, Approx={approx_jaccard:.6f}, Diff={diff:.6f}")
                consistency_checks += 1
    print(f"Performed {consistency_checks} consistency checks")

    for i in range(exact_data.shape[0]):
        # Get the indices of the first 30 and the last 30 values in the row
        first_indices = np.arange(min(30, exact_data.shape[1]))
        last_indices = np.arange(max(0, exact_data.shape[1] - 30), exact_data.shape[1])
        # selected_indices = np.unique(np.concatenate((first_indices, last_indices)))
        selected_indices = last_indices

        for j in selected_indices:
            exact_jaccard = exact_data[i, j] / (exact_data[i, i] + exact_data[j, j] - exact_data[i, j])
            approximate_jaccard = approximate_jaccard_data[i,j]
            if exact_data[i, i] * exact_data[j, j] > np.iinfo(np.int32).max:
                print(f"Overflow detected: sizes[{i}] * sizes[{j}] exceeds int32 limit.")
                continue
            exact_distances.append(exact_jaccard)
            approximate_distances.append(approximate_jaccard)
    

    print("loaded the matrix ")


    # Plot the results
    plt.figure(figsize=(8, 8))
    plt.scatter(exact_distances, approximate_distances, alpha=0.5, s=1, label="Pair of datasets")
    plt.plot([min(exact_distances), max(exact_distances)],
             [min(exact_distances), max(exact_distances)], color='red', linestyle='--', label="Identity line")
    # plt.plot([min(exact_distances), max(exact_distances)],
    #          [min(exact_distances) + 0.033, max(exact_distances) + 0.033], color='green', linestyle='--', label="jaccard + 0.033")
    # plt.plot([min(exact_distances), max(exact_distances)],
    #          [min(exact_distances) - 0.033, max(exact_distances) - 0.033], color='green', linestyle='--', label="jaccard - 0.033")
    x = np.linspace(min(exact_distances), max(exact_distances), 500)
    # uncertainty_minhash = 2.33 * np.sqrt(x * (1 - x) / 512)
    # plt.plot(x, x + uncertainty_minhash, color='purple', linestyle='--', label="Upper uncertainty line for 1024 MinHashes")
    # plt.plot(x, x - uncertainty_minhash, color='purple', linestyle='--', label="Lower uncertainty line for 1024 MinHashes")
    plt.xlabel("Exact Jaccard", fontsize=18)
    plt.ylabel("Estimated Jaccard", fontsize=18)
    # plt.title("Approximate vs Exact Jaccard on 35k examples")
    plt.xlim([0,1])
    plt.ylim([-0.05, 1.05])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute the differences between exact and approximate Jaccard values
    differences = np.array(exact_distances) - np.array(approximate_distances)

    # Filter out differences where exact_distance is 0 (which are distances between a dataset and itself)
    filtered_differences = [diff for diff, exact in zip(differences, exact_distances) if exact != 0]
    filtered_differences = np.array(filtered_differences)
    # Filter out NaN values in differences
    filtered_differences = filtered_differences[~np.isnan(filtered_differences)]

    # Plot a normalized histogram of the differences
    plt.figure(figsize=(8, 6))
    counts, bins, _ = plt.hist(filtered_differences, bins=500, density=True, alpha=0.6, color='blue', label='Differences')
    plt.xlabel("Exact Jaccard - Approximate Jaccard")
    plt.ylabel("Normalized Density")
    plt.title("Differences between exact and estimated Jaccard on 35k examples")
    plt.xlim([-0.1, 0.1])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Initialize lists to store metrics for each bin
    variances = []
    percentile_99 = []

    bins = [-0.0001, 0.0, 0.1, 0.15, 0.25, 0.35, 0.5, 0.999]

    # Compute metrics for each bin
    for b in range(len(bins) - 1):
        bin_mask = [(exact_distances[i] > bins[b]) and (exact_distances[i] <= bins[b + 1]) and not np.isnan(differences[i]) for i in range(len(exact_distances))]
        bin_differences = np.array([diff for diff, mask in zip(differences, bin_mask) if mask])

        if len(bin_differences) > 0:
            # Append standard deviation of the differences for this bin
            variances.append(float(np.std(bin_differences)))

            # Compute 99th percentile
            percentile_99.append(np.percentile(bin_differences, 95))

            # Compute quartiles and percentiles
            quartiles = np.percentile(bin_differences, [25, 50, 75])

        else:
            # If no data in the bin, append 0
            print("whajfm")
            sys.exit()

    bin_centers = [(bins[b] + bins[b + 1]) / 2 for b in range(len(bins) - 1)]

    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, variances, marker='o', label='Std dev')
    plt.plot(bin_centers, percentile_99, marker='o', label='95th Percentile 35k genome')
    plt.xlabel("Exact Jaccard Distance")
    plt.ylabel("Metric Value")
    plt.title("Distance of vectors approximation as a Function of Exact Jaccard Distance")
    plt.ylim([0, 0.04])
    plt.xlim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()


    #compute the variance we would have had for minHash
    minhash_variances = [np.sqrt(i*(1-i)/1600) for i in exact_distances]
    vars = []
    for b in range(len(bins) - 1):
        bin_mask = [(exact_distances[i] > bins[b]) and (exact_distances[i] <= bins[b + 1]) and not np.isnan(differences[i]) for i in range(len(exact_distances))]
        bin_var = np.array([diff for diff, mask in zip(minhash_variances, bin_mask) if mask])

        if len(bin_differences) > 0:
            # Append standard deviation of the differences for this bin
            vars.append(float(np.mean(bin_var)))

        else:
            # If no data in the bin, append 0
            print("whajfm")
            sys.exit()

    bin_centers = [(bins[b] + bins[b + 1]) / 2 for b in range(len(bins) - 1)]
    print("varas ", vars)

    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, vars, marker='o', label='Std dev')
    plt.xlabel("Exact Jaccard Distance")
    plt.ylabel("Metric Value")
    plt.title("Distance of vectors approximation as a Function of Exact Jaccard Distance")
    plt.ylim([0, 0.04])
    plt.xlim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_exact_and_random_proj():

    size_of_datasets = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/hash_counts.txt"
    sizes = []
    names = []
    with open(size_of_datasets, "r") as f:
        for line in f :
            sizes.append(int(line.strip().split()[1]))
            names.append(line.strip().split()[0])

    exact_matrix = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/exact_ava.bin"
    # Load the exact matrix
    with open(exact_matrix, "rb") as f:
        exact_data = np.frombuffer(f.read(), dtype=np.int32).reshape(35003, 35003)

    exact_distances = []
    approximate_distances = []

    # metagenome vector sketches
    approximate_matrix = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/ava_distances_dense.bin"
    # approximate_matrix = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/ava_distances_dense_20000.bin"
    # Load the approximate matrix
    with open(approximate_matrix, "rb") as f:
        approximate_data = np.frombuffer(f.read(), dtype=np.int32).reshape(35003, 35003)
        
    # Take the highest 30 values of each row and also pairs where exact Jaccard is 0
    for i in range(exact_data.shape[0]):
        # Get the indices of the first 30 and the last 30 values in the row
        first_indices = np.arange(min(30, exact_data.shape[1]))
        last_indices = np.arange(max(0, exact_data.shape[1] - 30), exact_data.shape[1])
        # selected_indices = np.unique(np.concatenate((first_indices, last_indices)))
        selected_indices = last_indices

        for j in selected_indices:
            exact_jaccard = exact_data[i, j] / (exact_data[i, i] + exact_data[j, j] - exact_data[i, j])
            approximate_jaccard = approximate_data[i, j] / (approximate_data[i, i] + approximate_data[j, j] - approximate_data[i, j])
            # approximate_jaccard = approximate_data[i,j] / 1e6
            if exact_data[i, i] * exact_data[j, j]> np.iinfo(np.int32).max:
                print(f"Overflow detected: sizes[{i}] * sizes[{j}] exceeds int32 limit.")
                continue
            if exact_jaccard == 0.75:
                print("Exact Jaccard of 0.75 between ", i, " ", j, " : ", approximate_jaccard)
            exact_distances.append(exact_jaccard)
            approximate_distances.append(approximate_jaccard)

    print("loaded the matrix ")


    # Plot the results
    plt.figure(figsize=(8, 8))
    plt.scatter(exact_distances, approximate_distances, alpha=0.5, s=1, label="Pair of datasets")
    plt.plot([min(exact_distances), max(exact_distances)],
             [min(exact_distances), max(exact_distances)], color='red', linestyle='--', label="Identity line")
    x = np.linspace(min(exact_distances), max(exact_distances), 500)
    # uncertainty_minhash = 2.33 * np.sqrt(x * (1 - x) / 512)
    # plt.plot(x, x + uncertainty_minhash, color='purple', linestyle='--', label="Upper uncertainty line for 1024 MinHashes")
    # plt.plot(x, x - uncertainty_minhash, color='purple', linestyle='--', label="Lower uncertainty line for 1024 MinHashes")
    plt.xlabel("Exact Jaccard", fontsize=18)
    plt.ylabel("Estimated Jaccard", fontsize=18)
    # plt.title("Approximate vs Exact Jaccard on 35k examples")
    plt.xlim([0,1])
    plt.ylim([-0.05, 1.05])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute the differences between exact and approximate Jaccard values
    differences = np.array(exact_distances) - np.array(approximate_distances)

    # Filter out differences where exact_distance is 0 (which are distances between a dataset and itself)
    filtered_differences = [diff for diff, exact in zip(differences, exact_distances) if exact != 0]
    filtered_differences = np.array(filtered_differences)
    # Filter out NaN values in differences
    filtered_differences = filtered_differences[~np.isnan(filtered_differences)]

    # Plot a normalized histogram of the differences
    plt.figure(figsize=(8, 6))
    counts, bins, _ = plt.hist(filtered_differences, bins=500, density=True, alpha=0.6, color='blue', label='Differences')
    plt.xlabel("Exact Jaccard - Approximate Jaccard")
    plt.ylabel("Normalized Density")
    plt.title("Differences between exact and estimated Jaccard on 35k examples")
    plt.xlim([-0.1, 0.1])
    plt.legend()
    plt.grid(True)
    plt.show()

    # Initialize lists to store metrics for each bin
    variances = []
    percentile_99 = []

    bins = [-0.0001, 0.0, 0.1, 0.15, 0.25, 0.35, 0.5, 1]

    percentile=95
    # Compute metrics for each bin
    for b in range(len(bins) - 1):
        bin_mask = [(exact_distances[i] > bins[b]) and (exact_distances[i] <= bins[b + 1]) and not np.isnan(differences[i]) for i in range(len(exact_distances))]
        bin_differences = np.array([diff for diff, mask in zip(differences, bin_mask) if mask])

        if len(bin_differences) > 0:
            # Append standard deviation of the differences for this bin
            variances.append(float(np.std(bin_differences)))

            # Compute 99th percentile
            percentile_99.append(np.percentile(bin_differences, percentile))

            # Compute quartiles and percentiles
            quartiles = np.percentile(bin_differences, [25, 50, 75])

        else:
            # If no data in the bin, append 0
            print("whajfm")
            sys.exit()

    bin_centers = [(bins[b] + bins[b + 1]) / 2 for b in range(len(bins) - 1)]
    print("values: ", variances)

    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, variances, marker='o', label='Std dev')
    plt.plot(bin_centers, percentile_99, marker='o', label=f'{percentile}th Percentile 35k genome')
    plt.xlabel("Exact Jaccard Distance")
    plt.ylabel("Metric Value")
    plt.title("Distance of vectors approximation as a Function of Exact Jaccard Distance")
    plt.ylim([0, 0.04])
    plt.xlim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_error_minhash():

    #binomial of parameter J 
    ...

def measure_error_theoretically():

    num_pairs = 1000

    dimension = 2048
    random.seed(0)
    np.random.seed(0)

    curr_id = 0
    errors = []
    exact_js = []
    approx_js = []
    dots = []

    j_values = np.linspace(0, 1, 30)  # 101 values from 0 to 1 (0%, 1%, ..., 100%)
    errors_by_j = {j: [] for j in j_values}

    true_jaccard = []
    approx_jaccard = []

    for i,j in enumerate(j_values):
        print("looking at ", j)
        for _ in range(1000):  # 1000 draws for each percentage of j
            # draw a common size for both sets
            s = random.randint(100, 10000)
            s1 = s2 = s

            # compute intersection size (rounded) and clamp to [0, s]
            j_2 = random.uniform(j_values[i], j_values[i+1] if i < len(j_values) - 1 else 1)
            inter = int(round(2.0 * s * j_2 / (1.0 + j_2)))
            inter = max(0, min(inter, s))

            # project the sets
            common_vec = get_me_a_random_projection_like_vector(dimension, inter)
            vecA = get_me_a_random_projection_like_vector(dimension, s1 - inter)
            vecA += common_vec
            vecB = get_me_a_random_projection_like_vector(dimension, s2 - inter)
            vecB += common_vec

            # dot product approximates intersection size in expectation
            dot = float(np.dot(vecA, vecB))
            est_A = float(np.dot(vecA, vecA))
            est_B = float(np.dot(vecB, vecB))

            # compute exact and estimated jaccard
            denom_exact = (s1 + s2 - inter)
            exact_j = (inter / denom_exact) if denom_exact != 0 else 0.0
            denom_est = (est_A + est_B - dot)
            est_j = (dot / denom_est) if denom_est != 0 else 0.0

            true_jaccard.append(exact_j)
            approx_jaccard.append(est_j)

            errors_by_j[j].append(exact_j - est_j)

    # Plot true vs approximate Jaccard for all pairs
    plt.figure(figsize=(8, 8))
    plt.scatter(true_jaccard, approx_jaccard, alpha=0.3, s=1)

    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label="Identity line")
    plt.xlabel("True Jaccard", fontsize=24)
    plt.ylabel("Estimated Jaccard", fontsize=24)
    plt.xlim([0, 1])
    plt.ylim([-0.5, 1.1])
    plt.legend()
    plt.grid(True)
    plt.show()

    # # Compute the first and 99th percentiles for each j
    # j_percentiles = []
    # first_percentiles = []
    # ninety_ninth_percentiles = []
    # std_deviations = []

    # for j, error_list in errors_by_j.items():
    #     error_array = np.array(error_list)
    #     j_percentiles.append(j)
    #     first_percentiles.append(np.percentile(error_array, 1))
    #     ninety_ninth_percentiles.append(np.percentile(error_array, 99))
    #     std_deviations.append(np.std(error_array))

    # # Plot the first and 99th percentiles as a function of j
    # plt.figure(figsize=(8, 6))
    # # plt.plot(j_percentiles, first_percentiles, label="1st Percentile vector sketches (theoretical)", color="blue")
    # plt.plot(j_percentiles, ninety_ninth_percentiles, label="99th Percentile vector sketches (theoretical)", color="blue")
    # plt.plot(j_percentiles, std_deviations, label="Std Deviation vector sketches (theoretical)", color="green")

    # # # Compute and plot the percentiles for MinHash sketches
    # # minhash_error = 2.36 * np.sqrt(np.array(j_percentiles) * (1 - np.array(j_percentiles)) * 0.25 / dimension*2)
    # # plt.plot(j_percentiles, -minhash_error, label="1st Percentile MinHash sketches", color="orange", linestyle="--")
    # # plt.plot(j_percentiles, minhash_error, label="99th Percentile MinHash sketches", color="orange", linestyle="--")

    # plt.xlabel("True jaccard (j)")
    # plt.ylabel("Difference between true and estimated Jaccard")
    # plt.title("Upper bound on errors of sketches as a Function of Jaccard\n2048 values")
    # plt.ylim([0, 0.04])
    # plt.legend()
    # plt.grid(True)
    # plt.show()

def measure_the_highest_coordinate_of_the_vectors():
    file = "~/Documents/postdoc/penn_state/pairwise_comp/faiss_db/big_vectors.bin"

    file = os.path.expanduser(file)
    # Read the file as int32 in a streaming fashion
    highest_value = None
    chunk_size = 1024 * 1024  # Read 1MB chunks

    with open(file, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data = np.frombuffer(chunk, dtype=np.int32)
            max_in_chunk = np.max(data)
            if highest_value is None or max_in_chunk > highest_value:
                highest_value = max_in_chunk

    print(f"The highest int in the file is: {highest_value}")

def check_16_bits_work_well():
    file_32_bits = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/faiss_db/vectors.bin"
    file_16_bits = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/all_vectors_16bits/vectors.bin"

    # Read and compare the two files
    with open(file_32_bits, "rb") as f32, open(file_16_bits, "rb") as f16:
        # Read both files completely
        data_32 = np.frombuffer(f32.read(), dtype=np.int32)
        data_16 = np.frombuffer(f16.read(), dtype=np.int16)
        
        # Reshape to vectors of 2048 dimensions
        vectors_32 = data_32.reshape(-1, 2048)
        vectors_16 = data_16.reshape(-1, 2048)
        
        print(f"32-bit file: {vectors_32.shape[0]} vectors")
        print(f"16-bit file: {vectors_16.shape[0]} vectors")
        
        if vectors_32.shape[0] != vectors_16.shape[0]:
            print("ERROR: Different number of vectors! ", vectors_32.shape[0], vectors_16.shape[0])
            return
        
        # Compare each vector
        all_match = True
        for i in range(vectors_32.shape[0]):
            if not np.array_equal(vectors_32[i], vectors_16[i].astype(np.int32)):
                print(f"Vector {i} differs!")
                all_match = False
                if i < 5:  # Show first few differences
                    diff_indices = np.where(vectors_32[i] != vectors_16[i].astype(np.int32))[0]
                    print(f"  Differences at indices: {diff_indices[:10]}")
        
        if all_match:
            print("All vectors match perfectly!")
        else:
            print("Some vectors differ between 32-bit and 16-bit files")

def plot_errors():

    std_dev_minhash = [0.0, 0.0023186762615425246, 0.008248126278362815, 0.009901403660995102, 0.01156192733484304, 0.012324822478635792, 0.011848462896191079]
    #[0.0, 0.00204943963489124, 0.007290382529391634, 0.008751687089943679, 0.010219396527567027, 0.010893706939454701, 0.010472660575667394]
    std_dev_dashing = [0.0, 0.0025163098743849434, 0.007937888453352894, 0.009796985531193564, 0.010618605330536744, 0.008823785679533687, 0.004757405672670887]
    std_dev_dothash = [0.005994300274383996, 0.008827218065569511, 0.011000138790249706, 0.009997441473670012, 0.009941740124741366, 0.012545220027072269, 0.00790386934646627]
    #[0.005490915524182698, 0.008269956761652023, 0.010169285305502191, 0.009201802766961763, 0.009047351383964185, 0.011573646141282864, 0.005615676212926416]
    x =[-5e-05, 0.05, 0.125, 0.2, 0.3, 0.425, 0.7495]

    plt.figure(figsize=(10, 6))
    plt.plot(x, std_dev_minhash, marker='o', label='MinHash, n=1600', color='red')
    plt.plot(x, std_dev_dashing, marker='s', label='HyperLogLog, S=2048', color='blue')
    plt.plot(x, std_dev_dothash, marker='^', label='DotHash, d=1800, ', color='green')
    plt.xlabel('Exact Jaccard Distance', fontsize=22)
    plt.ylabel('RMSE', fontsize=22)
    plt.ylim([0, 0.02])
    plt.xlim([0,1])
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # plot_error_random_proj()
    # compare_exact_and_random_proj()
    # compare_exact_and_dashing()
    # measure_error_theoretically()
    # measure_the_highest_coordinate_of_the_vectors()
    # check_16_bits_work_well()
    plot_errors()