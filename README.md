# 🧬 Metagenome Vector Sketches

This repository provides code for sketching genomic data using random projection to efficiently process and compare large metagenomic datasets.

## 🛠️ Installation Guide

Follow these steps to set up the necessary environment and build the executables.

### Clone the Repository

Clone the repository and its submodules recursively:
```Shell

git clone --recursive https://github.com/RolandFaure/metagenome_vector_sketches.git
cd metagenome_vector_sketches
git submodule update --init --recursive
```

### Set Up the Conda Environment

Create a new Conda environment named faiss_env and install the required dependencies, including FAISS for fast similarity search.
```Shell

conda create -n faiss_env python=3.12
conda activate faiss_env
conda install -c pytorch faiss-cpu
conda install -c conda-forge pybind11 scipy matplotlib pandas
```
### Build the Executables

Navigate back to the main directory, create a build folder, and compile the C++ code using cmake. This step generates all necessary executables inside the build folder.
```Shell

cd metagenome_vector_sketches
mkdir build
cd build
cmake -DPython_EXECUTABLE=$(which python) \
      -DPython_ROOT_DIR=$CONDA_PREFIX \
      -DPython_FIND_STRATEGY=LOCATION \
      ..
cmake --build . -j 8
```

## 🚀 Usage Examples

The following examples use data in the `test` folder. All compiled executables are located inside the `build` folder. **Running any executable without arguments will display its usage instructions.**

### Create Projected Vectors

Use `project_everything` to create projected vectors from fracminhash data. The output vectors will be stored in the specified index folder (`toy_index/`).
```Shell

cd test/
../build/project_everything toy toy_db/ -t 8 -d 2048 -s 0
```
### Create FAISS Index

After generating vectors, you can create a FAISS index for efficient search using the Python script `jaccard.py`.
```Shell

python3 ../src/jaccard.py index toy_db -t 8
```

### Compute Pairwise Comparison Matrix

The `pairwise_comp_optimized` executable computes the similarity matrix between all vectors.

To compute the matrix:
```Shell

../build/pairwise_comp_optimized --db toy_db/ --dimension 2048 --output_folder toy_index/ --max_memory_gb 12 --num_threads 8
```
Strategy Note: The default strategy is 0=random projections. You can use --strategy 1 for MinHashes.

### Query the Pairwise Matrix

The `query_pc_mat` executable allows you to query the computed similarity matrix.

```shell
Query Pairwise Comparison Matrix

Usage:
        ../build/query_pc_mat [--matrix <folder>] [--db <folder>] [--query_file <file>] [--top
                              <int>] [--batch_size <int>] [--write_to_file <file>] [--show_all]
                              [--help]

        ../build/query_pc_mat [--matrix <folder>] [--db <folder>] [--query_ids <ids>...] [--top
                              <int>] [--batch_size <int>] [--write_to_file <file>] [--show_all]
                              [--help]

        ../build/query_pc_mat [--matrix <folder>] [--db <folder>] [--row_file <row> [--col_file]
                              <col>] [--top <int>] [--batch_size <int>] [--write_to_file <file>]
                              [--show_all] [--help]

Options:
  --matrix  Folder containing the pairwise matrix files
  --db  Folder containing the matrix meta data
  --query_file     File containing query IDs (one per line)
  --query_ids      Query IDs as command line arguments (numeric indices or identifiers)
  --row_file     File containing query row IDs (one per line)
  --col_file     File containing query col IDs (one per line)
  --top           Number of top jaccard values to show [default 10]
  --batch_size    Number of queries to process per batch [default 100]
  --write_to_file  Where to save the output (expected format: *.csv/*.tsv/*.npy/*npz for row-col query. *.csv/*tsv/*txt for regular query).
  --show_all  Whether to show all neighbors instead of top N
  --help           Show this help message

```

#### Regular Query (Nearest Neighbors)

Query the matrix for neighbors of specific IDs listed in a file (`query_strs.txt`):
```Shell

../build/query_pc_mat --matrix toy_index --db toy_db/ --query_file query_strs.txt --write_to_file toy_neighbors.txt --show_all
```

This command outputs one file per query ID (e.g., `DRR000821_toy_neighbors.txt`) containing all neighbors, as `--show_all` is specified.

#### Sliced Matrix Query (Sub-matrix)

Query a slice of the matrix (a sub-matrix) defined by IDs in a row file and a column file:
```Shell

../build/query_pc_mat --matrix toy_index --db toy_db/  --row_file row_file.txt --col_file col_file.txt --write_to_file row_col.npy
```
```
Important Output Format Note:

    Sliced (Row-Col) Query: Output file must be *.csv, *.tsv, *.npy, or *.npz.

    Regular Query: Output file must be *.csv, *.tsv, or *.txt.
```

### Python Interface for Matrix Search

The `read_pc_mat.py` script provides a Python interface for searching the pairwise comparison matrix.

```shell
Usage: read_pc_mat.py [-h] --matrix MATRIX --db DB [--query_file QUERY_FILE] [--row_file ROW_FILE] [--col_file COL_FILE]

Pairwise Comparison Matrix Search

options:
  -h, --help            show this help message and exit
  --matrix MATRIX       Folder containing matrix data
  --db DB               Folder containing auxilary information of the matrix
  --query_file QUERY_FILE
                        File with query IDs (one ID per line)
  --row_file ROW_FILE   File containing row IDs (one ID per line)
  --col_file COL_FILE   File containing column IDs (one ID per line)
```

#### Regular Query (Python)

```Shell

python3 ../src/read_pc_mat.py --matrix toy_index --db toy_db/ --query_file query_strs.txt
```

#### Sliced Matrix Query (Python)

```Shell

python3 ../src/read_pc_mat.py --matrix toy_index --db toy_db/ --row_file row_file.txt --col_file col_file.txt
```