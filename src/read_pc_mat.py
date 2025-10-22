import sys
import read_pc_mat_module as rpc
import numpy as np
import time

class PC_Matrix:
    def query_ava_matrix(matrix_folder, query_file):
        results = rpc.query(matrix_folder, query_file)
        formatted_results = []
        for res in results:
            formatted_results.append({
                'id': res['id'],
                'neighbor_ids': np.array(res['neighbor_ids']),
                'jaccard_similarities': np.array(res['jaccard_similarities'])
            })
        return formatted_results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python read_pc_mat.py <matrix_folder> <query_file>")
        sys.exit(1)

    matrix_folder = sys.argv[1]
    query_file = sys.argv[2]
    start_time = time.perf_counter()
    results = PC_Matrix.query_ava_matrix(matrix_folder, query_file)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"Query completed in {elapsed:.6f} seconds.\n")
    for i, res in enumerate(results):
        print(f"Query {res['id']}: #Neighbors = {len(res['neighbor_ids'])}")
        neighbors_to_show = min(10, len(res['neighbor_ids']))
        print('Top {} neighbors:'.format(neighbors_to_show))
        print("Neighbor IDs:", res['neighbor_ids'][:neighbors_to_show])
        print("Jaccard Similarities:", res['jaccard_similarities'][:neighbors_to_show])
        print()