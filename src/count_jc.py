import matplotlib.pyplot as plt
import pandas as pd
import sys

def get_dict(matrix_path):
    jaccard_dict = {i:0 for i in range(256)}
    for i in range(1000):
        shard_folder = matrix_path + "/shard_"+str(i)
        with open(shard_folder+"/jaccard_list.txt") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            jc = int(line.strip())
            jaccard_dict[i] += jc

    return jaccard_dict

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


def plot_hist(val_dict, marker):
    fn = 'jaccard_count.txt'
    with open(fn, 'w') as f:
        for (key,val) in val_dict.items():
            f.write(f'{key} {val}\n')
    print(f'Count saved to {fn}')
    # val_list = list(val_dict.values())
    # jc_list = list(val_dict.keys())
    # fig,ax = plt.subplots(figsize=(10, 6))
    # ax.bar(jc_list, val_list,  edgecolor='black', color='skyblue')
    # # ax.hist(val_list, bins=256,  edgecolor='black', color='skyblue')
    # ax.set_xlabel('Jaccard Estimates')
    # ax.set_ylabel('Frequency (log-scale)')
    # ax.set_yscale('log')
    # # ax.set_title(f'Includes neighbors with Jaccard >= {cut_off}')
    # out_fn = f'jaccard_dist.pdf'
    # fig.savefig(out_fn, dpi=300)
    # print('Figure saved to: '+out_fn)
    
    

def main():
    matrix_path = sys.argv[1]
    # marker = sys.argv[2]
    marker = 'dummy'
    val_dict = get_dict(matrix_path)
    plot_hist(val_dict, marker)

main()
