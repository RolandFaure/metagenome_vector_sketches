import numpy as np
import random
import cProfile
import pstats
import io
import time
import pickle
import matplotlib.pyplot as plt

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
    print("sqfldj")
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

def jaccard_exact():
    matrix_1 = np.load('/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/measure_precision/references/bacteroides_fragilis/all_vs_all.npy')
    matrix_1000 = np.load('/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/measure_precision/references/bacteroides_fragilis/all_vs_all_1000.npy')

    # Flatten the matrices to get all pairwise comparisons
    flat_1 = matrix_1.flatten()
    flat_1000 = matrix_1000.flatten()

    # Calculate correlation coefficient
    correlation = np.corrcoef(flat_1, flat_1000)[0, 1]
    print(f"Correlation between matrix_1 and matrix_1000: {correlation}")

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(flat_1, flat_1000, alpha=0.5)
    plt.xlabel('Exact Jaccard', fontsize=14)
    plt.ylabel('FracMinHash Jaccard, scale=1000', fontsize=14)
    plt.title(f'Correlation between ')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.grid(True)
    plt.show()

    # Calculate the absolute error between exact and FracMinHash Jaccard
    absolute_errors = np.abs(flat_1 - flat_1000)

    # Create bins for exact Jaccard values
    jaccard_bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1
    bin_centers = (jaccard_bins[:-1] + jaccard_bins[1:]) / 2

    # Calculate mean absolute error for each bin
    mean_errors = []
    for i in range(len(jaccard_bins) - 1):
        mask = (flat_1 >= jaccard_bins[i]) & (flat_1 < jaccard_bins[i+1])
        if np.sum(mask) > 0:
            mean_errors.append(np.mean(absolute_errors[mask]))
        else:
            mean_errors.append(0)

    # Plot error vs exact Jaccard
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, mean_errors, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Exact Jaccard', fontsize=14)
    plt.ylabel('Mean Absolute Error', fontsize=14)
    plt.title('FracMinHash Error vs Exact Jaccard (scale=1000)', fontsize=14)
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, max(mean_errors) * 1.1])
    plt.show()

def plot_error_minhash():

    #binomial of parameter J 
    ...

def does_the_error_depends_significantly_on_size_of_datasets():

    nb_points = 5000
    nb_point_per_points = 100
    jaccard = 0
    dimension = 2048
    sampling = 1000
    size_limits = [10000, 10000000000]
    values = [] #(size1, size2, error fracminhash, error random projection)

    for idx in range(nb_points):
        size1 = int(np.exp(np.random.uniform(np.log(size_limits[0]), np.log(size_limits[1]))))
        size2 = int(np.exp(np.random.uniform(np.log(size_limits[0]), np.log(size_limits[1]))))

        if size1 < jaccard*size2 or size2 < jaccard*size1:
            continue

        intersection_size = int((size1+size2)*jaccard/(1+jaccard))

        #now we will measure the error fracminhash does
        square_error_fmh = 0
        square_error_rp = 0
        for idx2 in range(nb_point_per_points) :
            sample_size_intersection = np.random.binomial(intersection_size, 1/sampling)
            sample_size_symdiff_1 = np.random.binomial(size1 - intersection_size, 1/sampling)
            sample_size_symdiff_2 = np.random.binomial(size2 - intersection_size, 1/sampling)
            estimation_jaccard_fracminhash = sample_size_intersection / (sample_size_symdiff_1+sample_size_symdiff_2+sample_size_intersection)
            square_error_fmh += (estimation_jaccard_fracminhash-jaccard)**2/nb_point_per_points

            vector_interseciton = get_me_a_random_projection_like_vector(dimension, sample_size_intersection)
            vector_simdiff_1 = get_me_a_random_projection_like_vector(dimension, sample_size_symdiff_1)
            vector_simdiff_2 = get_me_a_random_projection_like_vector(dimension, sample_size_symdiff_2)
            vector1 = vector_interseciton + vector_simdiff_1
            vector2 = vector_interseciton + vector_simdiff_2
            dot_product = np.dot(vector1, vector2)
            estimation_jaccard_random_proj = dot_product / (np.dot(vector1,vector1)+np.dot(vector2,vector2) - dot_product)
            square_error_rp += (estimation_jaccard_random_proj-jaccard)**2 /nb_point_per_points
            # square_error_rp += (estimation_jaccard_random_proj-estimation_jaccard_fracminhash)**2

            # if size1 > 10000000 and size2 > 10000000 :
            #     print("siezs ", size1, size2, intersection_size, sample_size_intersection, \
            #           sample_size_symdiff_1, sample_size_symdiff_2, estimation_jaccard_fracminhash)


        values.append((size1, size2, np.sqrt(square_error_fmh), np.sqrt(square_error_rp)))

    sizes1 = [v[0] for v in values]
    sizes2 = [v[1] for v in values]
    errors = [v[2] for v in values]
    errors_rp = [v[3] for v in values]

    plt.figure(figsize=(8, 6))
    x_range = np.logspace(np.log10(size_limits[0]), np.log10(size_limits[1]), 100)
    plt.plot(x_range, jaccard * x_range, 'r--', linewidth=2, label=f'Limits of attainable space')
    plt.plot(jaccard * x_range, x_range, 'r--', linewidth=2,)
    plt.legend()
    scatter = plt.scatter(sizes1, sizes2, c=errors, alpha=0.6, cmap='viridis', vmin=0, vmax=0.03)
    plt.colorbar(scatter, label='RMSE of Jaccard')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Size 1', fontsize=14)
    plt.ylabel('Size 2', fontsize=14)
    plt.xlim(size_limits)
    plt.ylim(size_limits)
    plt.title(f'RMSE of Jaccard estimation, true Jaccard={jaccard}\nFracMinHash sampling {1/sampling}', fontsize=14)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    x_range = np.logspace(np.log10(size_limits[0]), np.log10(size_limits[1]), 100)
    plt.plot(x_range, jaccard * x_range, 'r--', linewidth=2, label=f'Limits of attainable space')
    plt.plot(jaccard * x_range, x_range, 'r--', linewidth=2,)
    plt.legend()
    scatter = plt.scatter(sizes1, sizes2, c=errors_rp, alpha=0.6, cmap='viridis', vmin=0, vmax=0.03)
    plt.colorbar(scatter, label='RMSE of Jaccard')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Size 1', fontsize=14)
    plt.ylabel('Size 2', fontsize=14)
    plt.xlim(size_limits)
    plt.ylim(size_limits)
    plt.title(f'RMSE of Jaccard estimation, true Jaccard={jaccard}\nFracMinHash sampling {1/sampling}, Dimension {dimension}', fontsize=14)
    plt.grid(True)
    plt.show()

    # Plot FracMinHash error as a function of size1*size2
    size_products = [v[0] * v[1] for v in values]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(size_products, errors, alpha=0.6)
    plt.xscale('log')
    plt.xlabel('Size1 × Size2', fontsize=14)
    plt.ylabel('RMSE of Jaccard', fontsize=14)
    plt.title(f'FracMinHash RMSE vs Size Product\nTrue Jaccard={jaccard}, Sampling={1/sampling}', fontsize=14)
    plt.grid(True)
    plt.show()

    # Plot random projection error as a function of size1/size2 ratio for size1*size2 > 10^14
    size_ratios = []
    size_mults = []
    filtered_errors_rp = []
    
    for v in values:
        size1, size2, _, error_rp = v
        if size1 * size2 > 1e14:
            ratio = size1 / size2
            size_ratios.append(ratio)
            filtered_errors_rp.append(error_rp)
            size_mults.append(size1*size2)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(size_ratios, filtered_errors_rp, c=size_mults, alpha=0.6, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(scatter, label='Size1 × Size2')
    plt.xscale('log')
    plt.xlabel('Size1 / Size2', fontsize=14)
    plt.ylabel('RMSE of Jaccard', fontsize=14)
    plt.title(f'Random Projection RMSE vs Size Ratio\nTrue Jaccard={jaccard}, Dimension={dimension}, Size1×Size2 > 10^14', fontsize=14)
    plt.grid(True)
    plt.show()

def compute_error_for_all_points_in_space():

    dimension = 2048
    sampling = 1000

    realistic_sizes = [10000,30000,100000,300000,1000000,3000000,10_000000,30_000000, 100_000000, 300_000000, 1_000_000000, 3_000_000000, 10_000_000000, 30_000_000000, 100_000_000000]
    test_jaccard = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    nb_point_per_points = 500
    all_errors = {} #associates (size1,size2,jaccard) with (RMSE, 1th centile, 5th centile, 50th centile, 95th and 99th)

    for size1 in realistic_sizes :
        for size2 in realistic_sizes:
            for jaccard in test_jaccard :

                intersection_size = int((size1+size2)*jaccard/(1+jaccard))
                if intersection_size > size1 or intersection_size > size2 :
                    continue

                #now we will measure the error fracminhash does
                square_error_fmh = 0
                square_error_rp = 0
                all_errors_here = []
                for idx2 in range(nb_point_per_points) :
                    sample_size_intersection = np.random.binomial(intersection_size, 1/sampling)
                    sample_size_symdiff_1 = np.random.binomial(size1 - intersection_size, 1/sampling)
                    sample_size_symdiff_2 = np.random.binomial(size2 - intersection_size, 1/sampling)
                    estimation_jaccard_fracminhash = sample_size_intersection / (sample_size_symdiff_1+sample_size_symdiff_2+sample_size_intersection)
                    square_error_fmh += (estimation_jaccard_fracminhash-jaccard)**2/nb_point_per_points

                    vector_interseciton = get_me_a_random_projection_like_vector(dimension, sample_size_intersection)
                    vector_simdiff_1 = get_me_a_random_projection_like_vector(dimension, sample_size_symdiff_1)
                    vector_simdiff_2 = get_me_a_random_projection_like_vector(dimension, sample_size_symdiff_2)
                    vector1 = vector_interseciton + vector_simdiff_1
                    vector2 = vector_interseciton + vector_simdiff_2
                    dot_product = np.dot(vector1, vector2)
                    estimation_jaccard_random_proj = dot_product / (np.dot(vector1,vector1)+np.dot(vector2,vector2) - dot_product)
                    square_error_rp += (estimation_jaccard_random_proj-jaccard)**2 /nb_point_per_points
                    all_errors_here.append(estimation_jaccard_random_proj-jaccard)
                
                all_errors_here_sorted = sorted(all_errors_here)
                all_errors[(size1, size2, jaccard)] = (
                    np.sqrt(square_error_rp),
                    all_errors_here_sorted[max(0,int(nb_point_per_points/100)-1)],  # 1st percentile (min)
                    all_errors_here_sorted[int(5*nb_point_per_points/100)-1],  # 5th percentile
                    all_errors_here_sorted[int(50*nb_point_per_points/100)-1],  # 50th percentile (median)
                    all_errors_here_sorted[int(95*nb_point_per_points/100)-1],  # 95th percentile
                    all_errors_here_sorted[int(100*nb_point_per_points/100)-1]   # 99th percentile (max)
                )

                print("completed ", len(all_errors), " out of ", len(test_jaccard)*len(realistic_sizes)**2)

    with open('all_errors.pkl', 'wb') as f:
        pickle.dump(all_errors, f)
                
                
    # Load the data if it exists
    with open('all_errors.pkl', 'rb') as f:
        all_errors = pickle.load(f)

    # Create heatmaps for all size1 values
    unique_size1 = sorted(set(size1 for (size1, _, _) in all_errors.keys()))
    
    for target_size1 in unique_size1:
        # Extract data for the target size1
        size2_values = []
        jaccard_values = []
        rmse_values = []

        for (size1, size2, jaccard), (rmse, p1, p5, p50, p95, p99) in all_errors.items():
            if size1 == target_size1:
                size2_values.append(size2)
                jaccard_values.append(jaccard)
                rmse_values.append(rmse)

        if len(size2_values) == 0:
            print(f"No data found for size1 = {target_size1}")
            continue
        
        # Create a matrix for the heatmap
        unique_size2 = sorted(set(size2_values))
        unique_jaccard = sorted(set(jaccard_values))
        
        rmse_matrix = np.full((len(unique_jaccard), len(unique_size2)), np.nan)
        
        for size2, jaccard, rmse in zip(size2_values, jaccard_values, rmse_values):
            i = unique_jaccard.index(jaccard)
            j = unique_size2.index(size2)
            rmse_matrix[i, j] = rmse
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(rmse_matrix, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='RMSE')
        
        # Set tick labels
        plt.xticks(range(len(unique_size2)), [f'{s:.0e}' for s in unique_size2], rotation=45, ha='right')
        plt.yticks(range(len(unique_jaccard)), [f'{j:.2f}' for j in unique_jaccard])
        
        plt.xlabel('Size2', fontsize=14)
        plt.ylabel('Jaccard', fontsize=14)
        plt.title(f'RMSE Heatmap for Size1 = {target_size1:,}\nDimension={dimension}, Sampling={1/sampling}', fontsize=14)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # jaccard_exact()
    # does_the_error_depends_significantly_on_size_of_datasets()
    compute_error_for_all_points_in_space()