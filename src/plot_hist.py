import matplotlib.pyplot as plt
import pandas as pd
import sys

def get_list(matrix_path):
    neighbor_list = []
    no_neighbors = 0
    one_neighbor = 0
    for i in range(1000):
        shard_folder = matrix_path + "/shard_"+str(i)
        with open(shard_folder+"/neighbor_count.txt") as f:
            lines = f.readlines()
        for line in lines:
            num_of_neighbors = int(line.strip())
            num_of_neighbors -= 1
            neighbor_list.append(num_of_neighbors)
            # if num_of_neighbors == 1:
            #     no_neighbors+=1
            # elif num_of_neighbors == 2:
            #     one_neighbor += 1
    
    # print("total accessions: ", len(neighbor_list))
    # print("Total non-zero entries: ",sum(neighbor_list))
    print(f"{len(neighbor_list)} {sum(neighbor_list)} ",end='')
    # print(f"No neighbors: {no_neighbors}")
    # print(f"Only one neighbor: {one_neighbor}")
    return neighbor_list

# def plot_hist(val_list, cut_off):
#     fig,ax = plt.subplots(figsize=(10, 6))
#     ax.hist(val_list, bins=100, edgecolor='black', color='skyblue')
#     ax.set_xlabel('Neighbor Count')
#     ax.set_ylabel('Frequency (log-scale)')
#     ax.set_yscale('log')
#     ax.set_title(f'Includes neighbors with Jaccard >= {cut_off}')
#     out_fn = f'neighbor_hist_filter_{cut_off}.pdf'
#     fig.savefig(out_fn, dpi=300)
#     print('Figure saved to: '+out_fn)


def plot_hist(val_list, marker):
    import statistics
    # Define bin counts
    count_0 = sum(v == 0 for v in val_list)
    count_0_10 = sum(0 < v <= 10 for v in val_list)
    count_10_100 = sum(10 < v <= 100 for v in val_list)
    count_100_1000 = sum(100 < v <= 1000 for v in val_list)
    count_1000_10000 = sum(1000 < v <= 10000 for v in val_list)
    count_10000_inf = sum(v > 10000 for v in val_list)

    mean_val = statistics.mean(val_list)
    median_val = statistics.median(val_list)

    # Histogram bin edges
    # bins = [-0.5, 0.5, 10.5, 100.5, max(val_list) + 1]

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.hist(val_list, bins=bins, edgecolor='black', color='skyblue')

    # ax.set_xticks([0, 5.5, 55.5, (100.5 + max(val_list) + 1) / 2])
    # ax.set_xticklabels(['0', '(0,10]', '(10,100]', '(100,∞)'])

    # ax.set_xlabel('Neighbor Count')
    # ax.set_ylabel('Frequency (log-scale)')
    # ax.set_yscale('log')
    # ax.set_title(f'Includes neighbors with Jaccard ≥ {cut_off}')

    # out_fn = f'neighbor_hist_filter_{cut_off}.pdf'
    # fig.savefig(out_fn, dpi=300)
    # print(f'Figure saved to: {out_fn}')

    # Save counts to CSV
    df = pd.DataFrame({
        "Bin": ["{0}", "(0,10]", "(10,100]", "(100,1000)", "(1000,10000)", "(10000,∞)", "Mean", "Median"],
        "Count": [count_0, count_0_10, count_10_100, count_100_1000, count_1000_10000, count_10000_inf, mean_val, median_val]
    })

    csv_fn = f'neighbor_hist_filter_{marker}_counts.csv'
    df.to_csv(csv_fn, index=False)

    # print(f"{count_0} {count_0_10} {count_10_100} {count_100_1000} {count_1000_10000} {count_10000_inf} {mean_val} {median_val}")
    print(f"{mean_val} {median_val}")
    # print(f'Counts saved to: {csv_fn}')

def main():
    matrix_path = sys.argv[1]
    # marker = sys.argv[2]
    marker = 'dummy'
    val_list = get_list(matrix_path)
    plot_hist(val_list, marker)

main()
