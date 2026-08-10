# 🧬 Metagenome Vector Sketches

This repository provides code for sketching genomic data using random projection to efficiently process and compare large metagenomic datasets.

## 🛠️ Installation Guide

Follow these steps to set up the necessary environment and build the executables.

### Setting up the Repository

Clone the repository and its submodules recursively:

```Shell
git clone --recursive https://github.com/RolandFaure/metagenome_vector_sketches.git
cd metagenome_vector_sketches
git submodule update --init --recursive
```

You can use conda to install the dependencies:

```shell
conda create -n mgs python=3.12
conda activate mgs
conda install -c conda-forge hdf5 h5py
```

### Build the Executables

Create a build folder, and compile the C++ code using cmake. This step generates all necessary executables inside the build folder.

```Shell
mkdir build
cd build
cmake ..      
make -j 8
```

## 🚀 Usage Examples

The following examples use the FracMinHash data (signature files) inside the `test/toy/` folder. All compiled executables are located inside the `build` folder.

> **Tip:** Running any executable without arguments displays the complete command-line help.

### Build the Vector Database

Use `project_everything` to create projected vectors from FracMinHash data. 

```shell
Project FracMinHash Signatures to Vectors
Usage:
  Convert mode:
    ./project_everything convert <signature_folder> <hash_file> [-t threads]
      signature_folder : Path to folder containing signature files
      hash_file        : Output hash file path
      -t, --threads    : Number of threads (default: 1)

  Sketch mode:
    ./project_everything sketch <hash_file> <db_folder> [-t threads] [-d dimension] 
      hash_file        : Input hash file path
      db_folder        : Output folder for generated vector and auxiliary files
      -t, --threads    : Number of threads (default: 1)
      -d, --dimension  : Vector dimension (default: 2048)
  Convert & Sketch mode:
    ./project_everything build <signature_folder> <db_folder> [-t threads] [-d dimension]
      signature_folder : Path to folder containing signature files
      db_folder        : Output folder for generated vector and auxiliary files
      -t, --threads    : Number of threads (default: 1)
      -d, --dimension  : Vector dimension (default: 2048)
```

For example, to create and store vectors from the FracMinHash signature files inside the `test/toy/` folder to the folder (`toy_db/`):

```Shell
cd test/
../build/project_everything build toy toy_db/ -t 8 -d 2048
```

### Compute Pairwise Comparison Matrix

The `pairwise_comp_optimized` executable computes the similarity matrix among all vectors using the folder created from `project_everything` executable.

```shell
Create Pairwise Comparison Matrix

Usage:
        ./pairwise_comp_optimized --db <folder> --output_folder <folder> [--num_shards <int>]
                                  [--max_memory_gb <float>] [--num_threads <int>] [--help]

Options:
  --db              Folder containing the matrix meta data [Required]
  --output_folder   Folder where to store the matrix [Required]
  --num_threads     Numer of threads to use [default 1]
  --num_shards      Number of shards to use [default 1]
  --max_memory_gb   Max memory to be used per thread [default 1 GB]
  --help            Show this help message
```

For example, using the vector data inside the constructed `toy_db/` folder, one can create similarity matrix inside `toy_matrix` folder using:

```Shell
../build/pairwise_comp_optimized --db toy_db/ --output_folder toy_matrix/ --num_threads 2 --num_shards 4  --max_memory_gb 12 
```

### Query the Pairwise Matrix

The `query_pc_mat` executable allows you to query the computed similarity matrix.

```Shell
Query Pairwise Comparison Matrix

Usage:
        ./query_pc_mat --matrix <folder> --db <folder> [--query_file <file>] [--top <int>] [--thread
                       <int>] [--batch_size <int>] [--write_to_file <file>] [--show_all] [--print]
                       [--help]

        ./query_pc_mat --matrix <folder> --db <folder> [--query_ids <ids>...] [--top <int>]
                       [--thread <int>] [--batch_size <int>] [--write_to_file <file>] [--show_all]
                       [--print] [--help]

        ./query_pc_mat --matrix <folder> --db <folder> [--row_file <row> [--col_file] <col>] [--top
                       <int>] [--thread <int>] [--batch_size <int>] [--write_to_file <file>]
                       [--show_all] [--print] [--help]

        ./query_pc_mat --matrix <folder> --db <folder> [--filter <double> [--out] <folder>] [--top
                       <int>] [--thread <int>] [--batch_size <int>] [--write_to_file <file>]
                       [--show_all] [--print] [--help]

Options:
  --matrix        : Folder containing the pairwise matrix files [Required]
  --db            : Folder containing the matrix meta data [Required]
  --query_file    : File containing query IDs (one per line)
  --query_ids     : Query IDs as command line arguments (identifiers separated by space)
  --row_file      : File containing query row IDs (one per line)
  --col_file      : File containing query col IDs (one per line)
  --filter        : Filter values below threshold from matrix
  --out           : Output folder for the filtered matrix
  --top           : Number of top jaccard values to show [default 10]
  --batch_size    : Number of queries to process per batch [default 1000]
  --thread        : Number of threads to use [default 1]
  --write_to_file : Where to save the output. Expected format: 
                    - *.csv/*tsv/*txt for regular query.
                    - *.csv/*.tsv/*.npy/*npz/*h5 for row-col query.
  --show_all      : Whether to show all neighbors instead of top N
  --print         : Whether to print the outputs to screen
  --help          : Show this help message

```

> **Note**: Batches are executed in parallel, up to the configured number of threads. Within each batch, queries are processed sequentially. The write phase for sliced queries is also performed sequentially.

> **To query from all accessions inside the server, use `--matrix /scratch/mgs_project/matrix/ --db /scratch/mgs_project/db/`**

Inside the `test` folder, there are three example query files (`query_samples.txt`, `row_samples.txt` and `col_samples.txt`) that will be used for the following examples.
Three different kinds of queries are supported:

#### Regular Query (Nearest Neighbors)

Query the constructed matrix, `toy_matrix` for neighbors of specific IDs listed in a file (`query_samples.txt`):

```Shell
../build/query_pc_mat --matrix toy_matrix --db toy_db/ --query_file query_samples.txt --write_to_file toy_neighbors.txt --batch_size 5 --thread 2 --show_all
```

This command outputs one file per query ID (e.g., `DRR000821_toy_neighbors.txt`) containing all neighbors, as `--show_all` is specified.

#### Sliced Matrix Query (Sub-matrix)

Create a slice of the matrix (a sub-matrix) from specificed IDs in a row file (`row_samples.txt`) and a column file (`col_samples.txt`):

```Shell
../build/query_pc_mat --matrix toy_matrix --db toy_db/  --row_file row_samples.txt --col_file col_samples.txt --write_to_file row_col.h5 --batch_size 5 --thread 2
```

Here, use `*.h5` ([HDF5](https://www.hdfgroup.org/solutions/hdf5/)) as the output format to get the most compressed output. This format can be accessed conveniently in Python using the [h5py](https://docs.h5py.org/en/stable/) library.

#### Filter Matrix

Filter all accessions below a threshold from [0, 1] and write the corresponding matrix to a new location:

```Shell
../build/query_pc_mat --matrix toy_matrix --db toy_db/ --filter 0.2 --out filtered_toy_matrix --thread 2
```

```
Important Output Format Note:
    Regular Query: Output file must be *.csv, *.tsv, or *.txt.

    Sliced (Row-Col) Query: Output file must be *.csv, *.tsv, *.npy, *npz or *h5. *h5 gives the most compressed output.
```
