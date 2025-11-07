import sys
import read_pc_mat_module as rpc
import numpy as np
import time

class PC_Matrix:
    def query_ava_matrix(matrix_folder, query_file):
        start_time = time.perf_counter()
        results = rpc.query(matrix_folder, query_file)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        print(f"Query completed in {elapsed:.6f} seconds.\n")
        
        # modify formatting whicever way necessary
        formatted_results = []
        for res in results:
            formatted_results.append({
                'id': res['id'],
                'neighbor_ids': np.array(res['neighbor_ids']),
                'jaccard_similarities': np.array(res['jaccard_similarities'])
            })
        return formatted_results
    
    def query_pc_mat_sliced(matrix_folder, row_file, col_file):
        start_time = time.perf_counter()
        results = rpc.query_sliced(matrix_folder, row_file, col_file)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        print(f"Query completed in {elapsed:.6f} seconds.\n")

        # modify formatting whicever way necessary
        formatted_results = {
            'row_list': np.array(results['row-list']),
            'col_list': np.array(results['col-list']),
            'jac_dict': results['jac-dict']
        }
        return formatted_results

def process_query_file(matrix_folder, query_file):
    print(f"Processing query_file: {query_file} in {matrix_folder}")
    
    results = PC_Matrix.query_ava_matrix(matrix_folder, query_file)
    for i, res in enumerate(results):
        print(f"Query {res['id']}: #Neighbors = {len(res['neighbor_ids'])}")
        neighbors_to_show = min(10, len(res['neighbor_ids']))
        print('Top {} neighbors:'.format(neighbors_to_show))
        print("Neighbor IDs:", res['neighbor_ids'][:neighbors_to_show])
        print("Jaccard Similarities:", res['jaccard_similarities'][:neighbors_to_show])
        print()
    
def process_row_col(matrix_folder, row_file, col_file):
    print(f"Processing row_file: {row_file}, col_file: {col_file} in {matrix_folder}")
    results = PC_Matrix.query_pc_mat_sliced(matrix_folder, row_file, col_file)
    # print('Accession', end='\t')
    data = []
    for row in results['row_list']:
        data.append(results['jac_dict'][row])
    
    import pandas as pd
    df = pd.DataFrame(data, index=results['row_list'], columns=results['col_list'])
    print(df.to_string())
    


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pairwise Comparison Matrix Search"
    )

    parser.add_argument(
        "--matrix_folder",
        required=True,
        help="Folder containing matrix data"
    )

    # Mutually exclusive group does not allow enforcing *two args together*
    # So we'll add them manually and validate combination
    parser.add_argument("--query_file", help="File with query IDs (one ID per line)")
    parser.add_argument("--row_file", help="File containing row IDs (one ID per line)")
    parser.add_argument("--col_file", help="File containing column IDs (one ID per line)")

    args = parser.parse_args()

    # Validate choices
    if args.query_file:
        if args.row_file or args.col_file:
            parser.error("Cannot combine --query_file with --row_file/--col_file")
        process_query_file(args.matrix_folder, args.query_file)
    elif args.row_file and args.col_file:
        process_row_col(args.matrix_folder, args.row_file, args.col_file)
    else:
        parser.error("Must provide either --query_file or both --row_file AND --col_file")


if __name__ == "__main__":
    main()    
    
    
    