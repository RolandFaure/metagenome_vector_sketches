# metagenome_vector_sketches
Repository with code to sketch genomic data with random projection

## Installation

``` shell
git clone --recursive https://github.com/RolandFaure/metagenome_vector_sketches.git
cd metagenome_vector_sketches
git submodule update --init --recursive

conda create -n faiss_env python=3.12
conda activate faiss_env
conda install -c pytorch faiss-cpu
conda install -c conda-forge pybind11 scipy matplotlib pandas

cd metagenome_vector_sketches
mkdir build
cd build
cmake -DPython_EXECUTABLE=$(which python) \
      -DPython_ROOT_DIR=$CONDA_PREFIX \
      -DPython_FIND_STRATEGY=LOCATION \
      ..
cmake --build . -j 8
```


## Usage

We will use `test` folder for the example. **All executables are inside the `build` folder, and show usage when run without arguments.**

Create projected vectors from fracminhash data into the index folder:

```shell
cd test/
../build/project_everything toy toy_index/ -t 8 -d 2048 -s 0
```

Use the vectors to create FAISS index:

``` shell
python3 ../src/jaccard.py index toy_index -t 8
```

Use the vectors to create pairwise matrix:

``` shell
../build/pairwise_comp_optimized --vectors toy_index/vectors.bin --dimension 2048 --output_folder toy_index/ --max_memory_gb 12 --num_threads 8
```

../build/pairwise_comp_optimized --vectors toy_index_rice/vectors.bin --dimension 2048 --output_folder toy_index_rice/ --max_memory_gb 12 --num_threads 8

Then, to query using `query_pc_mat`:

<!-- ``` shell
../build/query_ava_matrix --matrix_folder toy_index/ --query_file query_strs.txt
``` -->
``` shell
../build/query_pc_mat --matrix_folder toy_index/ --query_file query_strs.txt --top 20 --write_to_file --show_all
```
If you want to use a row and col file for getting a sliced matrix:

``` shell
../build/query_pc_mat --matrix_folder toy_index/ --row_file row_file.txt --col_file col_file.txt --write_to_file
```


``` shell
../build/query_pc_mat --matrix_folder toy_index_rice/ --query_file query_strs.txt
```

To use python interface:

```shell
python3 ../src/read_pc_mat.py --matrix_folder toy_index --query_file query_strs.txt

```
Or to use a row and col file:

```shell
python3 ../src/read_pc_mat.py --matrix_folder toy_index --row_file row_file.txt --col_file col_file.txt

```

```
Usage

Pairwise Comparison Matrix Search

options:
  -h, --help            show this help message and exit
  --matrix_folder MATRIX_FOLDER
                        Folder containing matrix data
  --query_file QUERY_FILE
                        File with query IDs (one ID per line)
  --row_file ROW_FILE   File containing row IDs (one ID per line)
  --col_file COL_FILE   File containing column IDs (one ID per line)
```