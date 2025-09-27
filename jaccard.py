import faiss
import numpy as np
import scipy
import os
import argparse
import glob
import time
import shutil
import matplotlib.pyplot as plt
import random
import sys
import tempfile
import subprocess

__version__ = "1.0.2"
__date__ = "27/09/2025"

def index_vectors(output_dir):
    vectors = []

    # Ensure output_dir exists and contains only vectors.txt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        files = os.listdir(output_dir)
        allowed = {"vectors.bin", "vector_norms.txt", "dimension.txt"}
        for f in files:
            if f not in allowed:
                os.remove(os.path.join(output_dir, f))
    
    output_index = os.path.join(output_dir, "faiss.index")
    input_bin_vectors = os.path.join(output_dir, "vectors.bin")
    dim_name = os.path.join(output_dir, "dimension.txt")
    f = open(dim_name)
    dimension = int(f.readline().strip())
    f.close()

    with open(input_bin_vectors, "rb") as f:
        while True:
            bytes_read = f.read(4 * dimension)
            if not bytes_read or len(bytes_read) < 4 * dimension:
                break
            vec = np.frombuffer(bytes_read, dtype=np.int32)
            vectors.append(vec)

    vectors = np.stack(vectors).astype("float32")
    dim = vectors.shape[1]

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)  # Use inner product metric
    index.add(vectors)
    faiss.write_index(index, output_index)
    print(f"Indexed {len(vectors)} vectors of dimension {dim} into {output_index}.")

def search_index(index_folder, query_file, path_exec, j):

    dimension_file = os.path.join(index_folder, "dimension.txt")
    vectors_name_file = os.path.join(index_folder, "vector_norms.txt")
    with open(dimension_file, "r") as f:
        dimension = int(f.readline().strip())

    # Convert each query (hash list) into a random projected vector
    queries = []
    with open(query_file, "r") as f:

        # Prepare all queries for batch projection
        hash_lines = []
        sample_names = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 2:
                print("ERROR 332: ", query_file, " ", line[:20], " ", len(parts))
                sys.exit(332)
            sample_name = parts[0].strip()
            hash_list = parts[1].strip().split()
            hash_lines.append(" ".join(hash_list))
            sample_names.append(sample_name)

        # Write all queries to a temporary file
        with open(path_exec+"/tmp.hashes", mode="w") as tmp_hash_file:
            for hash_line in hash_lines:
                tmp_hash_file.write(f"{hash_line}\n")
            tmp_hash_file_path = tmp_hash_file.name

        # Call ./standalone_projection <tmp_file> <nb_dim>
        # print("running ", path_exec + "/standalone_projection", tmp_hash_file_path, str(dimension))
        result = subprocess.run(
            [path_exec + "/standalone_projection", tmp_hash_file_path, str(dimension)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"Error running standalone_projection: {result.stderr}")
            sys.exit(1)

        # Parse the output vectors, one per line
        output_lines = result.stdout.strip().splitlines()
        if len(output_lines) != len(hash_lines):
            print("ERROR 333: Number of output vectors does not match number of queries. ", len(hash_lines), len(output_lines))
            sys.exit(333)
        os.remove(tmp_hash_file_path)

        for vector_str in output_lines:
            vector = np.fromstring(vector_str, sep=" ")
            vector = vector / np.sqrt(dimension)
            queries.append(vector)

    index = faiss.read_index(index_folder+"faiss.index")

    queries = np.stack(queries).astype("float32")
    query_norms = np.linalg.norm(queries, axis=1)
    faiss.normalize_L2(queries)
    minimum_required_inner_product = 2*j/(1+j)
    
    # Start with a reasonable number of neighbors, increase until all above threshold are found
    nb_searches = 10
    found_all = False
    while not found_all:
        D, I = index.search(queries, nb_searches)
        # Check if all neighbors above threshold are included for each query
        found_all = True
        for i in range(D.shape[0]):
            if np.any(D[i] > minimum_required_inner_product):
                # If there are neighbors above threshold, check if the last returned is still above threshold
                if D[i][-1] > minimum_required_inner_product:
                    # Might need to fetch more neighbors
                    found_all = False
                    break
        if not found_all:
            nb_searches *= 2  # Double the number of neighbors and try again

    # Collect unique indices of neighbors to recover, flatten and filter out -1 (invalid)
    indices_to_recover = set()
    for neighbors in I:
        for idx in neighbors:
            if idx >= 0:
                indices_to_recover.add(idx)
    
    vectors_name_file = os.path.join(index_folder, "vector_norms.txt")
    vectors = {} #associates index -> (name, norm)
    with open(vectors_name_file, "r") as vec_nf:
        for idx, line in enumerate(vec_nf):
            if idx not in indices_to_recover:
                continue
            line = line.strip()
            if not line:
                print("ERROR 455959")
                sys.exit(23)
            name = line.split()[0]
            norm = float(line.split()[1])
            vectors[idx] = (name, norm, line)

    return_res = []
    for i, neighbors in enumerate(I):
        results = []
        for rank, idx in enumerate(neighbors):
            neighbor_id = vectors[idx][0]
            neighbor_norm = vectors[idx][1]
            query_norm = query_norms[i]
            inner_product = D[i, rank]

            jaccard_index = inner_product * query_norm * neighbor_norm / (neighbor_norm**2 + query_norm**2 - inner_product * query_norm * neighbor_norm)

            if jaccard_index > j:
                results.append((neighbor_id, jaccard_index, inner_product, neighbor_norm, query_norm))

        # Sort results by jaccard_index descending
        results.sort(key=lambda x: x[1], reverse=True)

        print(f"Query {i}:")
        for rank, (neighbor_id, jaccard_index, inner_product, neighbor_norm, query_norm) in enumerate(results):
            print(f"  Neighbor {rank}: {neighbor_id} (jaccard: {jaccard_index:.4f}), inner_product: {inner_product:.4f} {neighbor_norm} {query_norm}")
            return_res.append((i, neighbor_id, jaccard_index))

    return return_res

def test():

    index_folder = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/faiss_db"

    vectors_file = os.path.join(index_folder, "vector_norms.txt")
    ids = []
    with open(vectors_file, "r") as f:
        dimension = int(f.readline().strip())
        for line in f:
            ids.append(line.strip().split()[0])
    num_vectors = len(ids)
        
    print(f"Number of vectors in {vectors_file}: {num_vectors}")

    # Draw 20 random vector indices
    random_samples = set([ids[i] for i in random.sample(range(num_vectors), 20)])
    # Load all hashes from "hashes.txt"
    hashes_file = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/all_hashes.txt"
    query_file = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/test_query.txt"
    hashes_dict = {}
    with open(hashes_file, "r") as f, open(query_file, "w") as fq:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 2:
                continue
            sample_name = parts[0].strip()
            if sample_name in random_samples:
                hash_list = parts[1].strip().split()
                hashes_dict[sample_name] = hash_list
                fq.write(line+ "\n")

    print(f"Loaded hashes for {len(hashes_dict)} samples from {hashes_file}")

    output_file = os.path.join(index_folder, "random_neighbors.txt")
    results = []
    all_necessary_vectors = set()

    # Use search_index to get neighbors with jaccard > 0.05
    neighbors = search_index(index_folder + "/", query_file, os.path.dirname(os.path.abspath(__file__)), 0.05)
    for query_idx, neighbor_id, jaccard in neighbors:
        query_id = list(random_samples)[query_idx]
        results.append((query_id, neighbor_id, jaccard))
        all_necessary_vectors.add(neighbor_id)
    os.remove(query_file)

    print("queried the 20 vectors")

    # Load all hashes from "hashes.txt"
    hashes_dict = {}
    if os.path.exists(hashes_file):
        with open(hashes_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                sample_name = parts[0].strip()
                if sample_name in all_necessary_vectors:
                    hash_list = parts[1].strip().split()
                    hashes_dict[sample_name] = hash_list
        print(f"Loaded hashes for {len(hashes_dict)} samples from {hashes_file}")
    else:
        print(f"Hashes file {hashes_file} not found.")


    # For every pair with jaccard > 0.05, compute their actual jaccard from the hashes
    actual_jaccard_results = []
    for query_id, neighbor_id, jaccard in results:
        hashes1 = set(hashes_dict.get(query_id, []))
        hashes2 = set(hashes_dict.get(neighbor_id, []))
        if not hashes1 or not hashes2:
            continue
        intersection = hashes1 & hashes2
        union = hashes1 | hashes2
        if union:
            actual_jaccard = len(intersection) / len(union)
            actual_jaccard_results.append((query_id, neighbor_id, jaccard, actual_jaccard))

    # Optionally print or save the results
    for query_id, neighbor_id, jaccard, actual_jaccard in actual_jaccard_results:
        print(f"{query_id} vs {neighbor_id}: vector_jaccard={jaccard:.4f}, hash_jaccard={actual_jaccard:.4f}")

    # Scatter plot: x = true jaccard, y = estimated jaccard
    xs = [actual_jaccard for _, _, _, actual_jaccard in actual_jaccard_results]
    ys = [jaccard for _, _, jaccard, _ in actual_jaccard_results]
    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, alpha=0.1)
    # Plot x=y line
    min_val = min(xs + ys)
    max_val = max(xs + ys)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='x = y')
    plt.xlabel("True Jaccard")
    plt.ylabel("Estimated Jaccard")
    plt.legend()
    plt.show()

def main():

    # test()
    # sys.exit()

    parser = argparse.ArgumentParser(description="FAISS indexer and searcher.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_index = subparsers.add_parser("index", help="Index vectors from a file.")
    parser_index.add_argument("output_index", type=str, help="Path to output FAISS index folder.")

    parser_search = subparsers.add_parser("search", help="Search vectors in a FAISS index.")
    parser_search.add_argument("index_folder", type=str, help="Path to FAISS index folder.")
    parser_search.add_argument("query_file", type=str, help="Path to query file. Formatted as ID: space_separated_hashes, one ID per line per line")
    parser_search.add_argument("-j", type=float, default=0.1, help="Retrieve all datasets with higher Jaccard index")

    parser.add_argument("-v", "--version", action="store_true", help="Show version and date")

    args = parser.parse_args()

    if args.version:
        print(f"Version: {__version__}, Date: {__date__}")
        sys.exit(0)

    if args.command == "index":
        index_vectors(args.output_index)
    elif args.command == "search":
        if args.index_folder[-1] != "/":
            args.index_folder += "/"
        search_index(args.index_folder, args.query_file, os.path.dirname(os.path.abspath(__file__)), args.j)

if __name__ == "__main__":
    main()