# metagenome_vector_sketches
Repository with code to sketch genomic data with random projection

## Installation

``` shell
git clone --recursive https://github.com/RolandFaure/metagenome_vector_sketches.git
git submodule update --init --recursive

conda create -n faiss_env python=3.12
conda activate faiss_env
conda install -c pytorch faiss-cpu
conda install -c conda-forge pybind11 scipy matplotlib

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

We will use `test` folder for the example. All executables are in the `build` folder, and shows usage when run without arguments.

Create projected vectors from fracminhash data into the index folder:

```shell
cd test/
../build/project_everything toy toy_index/ -t 8 -d 1024 -s 0
```

Use the vectors to create FAISS index:

``` shell
python3 ../src/jaccard.py index toy_index -t 8
```

Use the vectors to create pairwise matrix:

``` shell
../build/pairwise_comp_optimized --vectors toy_index/vectors.bin --dimension 1024 --output_folder toy_index/ --max_memory_gb 12 --num_threads 8
```

Then, to query using `query_pc_mat`:

<!-- ``` shell
../build/query_ava_matrix --matrix_folder toy_index/ --query_file query_ids.txt
``` -->
``` shell
../build/query_pc_mat --matrix_folder toy_index/ --query_file query_ids.txt
```

To use python interface:

```shell
python3 ../src/read_pc_mat.py toy_index query_ids.txt

```
