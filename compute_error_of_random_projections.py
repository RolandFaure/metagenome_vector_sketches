import numpy as np
import random
import cProfile
import pstats
import io
import time
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


