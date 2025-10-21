import sys
import read_pc_mat_module as rpc
import numpy as np

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

    results = PC_Matrix.query_ava_matrix(matrix_folder, query_file)
    for i, res in enumerate(results):
        print(f"Query {res['id']}:")
        print("Neighbor IDs:", res['neighbor_ids'])
        print("Jaccard Similarities:", res['jaccard_similarities'])
        print()