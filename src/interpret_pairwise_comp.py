import argparse
import os
import sys
import itertools
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI tool to process distance matrices in a folder."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing distance matrices."
    )
    return parser.parse_args()

def load_matrices(folder):

    os.system("zstd -d " + folder + "/*.zst")

    #load the row_index.txt file
    address_of_rows = {}
    list_of_rows = []
    with open(folder+"/row_index.txt") as f:
        for line in f:
            ls = line.strip().split()
            address_of_rows[int(ls[0])] = int(ls[1])
            list_of_rows.append(int(ls[0]))

    neighbors = {}
    with open(folder+"/matrix.bin", "rb") as f :
        size_of_file = os.path.getsize(folder + "/matrix.bin")
        for r, row in enumerate(list_of_rows):

            # if row != 10:
            #     continue

            if r < len(list_of_rows) -1 :
                number_of_neighbors = (address_of_rows[list_of_rows[r+1]] - address_of_rows[list_of_rows[r]]) / 8
            else : 
                number_of_neighbors = (size_of_file - address_of_rows[list_of_rows[r]]) / 8
            f.seek(address_of_rows[row])               

            neighbor_differences = list(int.from_bytes(f.read(4), "little") for _ in range(int(number_of_neighbors)))
            neighbor_indices = list(itertools.accumulate(neighbor_differences))
            neighbor_values = list(int.from_bytes(f.read(4), "little") for _ in range(int(number_of_neighbors)))

            if row == 10:
                print("For row 10: ")
                print("for qiof ds ", number_of_neighbors, " ", len(neighbor_differences))
                print(neighbor_indices)
                print(neighbor_values)
            neighbors[row] = (neighbor_indices, neighbor_values)

    return neighbors

def main():
    args = parse_args()
    input_folder = args.input_folder

    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Placeholder: Load and process distance matrices here
    neighbors = load_matrices(input_folder)
    names = []
    norms = []

    with open(input_folder+"/vector_norms.txt") as f :
        index = 0
        for line in f:
            names.append(line.split()[0])
            norms.append(float(line.strip().split()[1]))
            index += 1

    neighbor_indices, neighbor_values = neighbors[10]
    pairs = list(zip(neighbor_indices, neighbor_values))
    pairs_with_jaccard = [
        (idx, val, val / (norms[10]**2 + norms[idx]**2 - val)) for idx, val in pairs
    ]
    pairs_sorted = sorted(pairs_with_jaccard, key=lambda x: x[2], reverse=True)

    for idx, val, jaccard in pairs_sorted:
        if jaccard > 0.05 or True:
            print(f"Neighbor index: {names[idx]}, Value: {val}, Jaccard: {jaccard:.4f}")
    
    num_neighbors = [len(neighbors[row][0]) for row in neighbors]
    plt.hist(num_neighbors, bins=30, edgecolor='black')
    plt.xlabel('Number of neighbors per sample')
    plt.ylabel('Number of accessions')
    plt.title('Histogram of Number of Neighbors per Sample')
    plt.show()

if __name__ == "__main__":
    main()