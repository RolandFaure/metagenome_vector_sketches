#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <Eigen/Dense>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <filesystem>

#include "clipp.h"
    
namespace fs = std::filesystem;
using namespace Eigen;
using namespace std;

struct SparseResult {
    vector<int> rows;
    vector<int> cols;
    vector<int64_t> values;
};

// Load a block of vectors from binary file
MatrixXi load_matrix_block(const string& file_path, int dimension, int begin, int end) {
    ifstream file(file_path, ios::binary);
    if (!file) {
        cerr << "Error opening file: " << file_path << endl;
        return MatrixXi();
    }
    
    uint64_t vector_size = dimension * sizeof(int32_t);
    file.seekg(begin * vector_size);
    
    int num_vectors = end - begin;
    vector<int32_t> buffer(num_vectors * dimension);
    file.read(reinterpret_cast<char*>(buffer.data()), num_vectors * vector_size);
    
    MatrixXi matrix(dimension, num_vectors);
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < dimension; ++j) {
            matrix(j, i) = buffer[i * dimension + j];
        }
    }
    
    return matrix;
}

// Optimized sparse dot product computation with early threshold checking
SparseResult compute_sparse_dot_products_optimized(
    const MatrixXi& block_i, 
    const MatrixXi& block_j, 
    const VectorXd& norms_i, 
    const VectorXd& norms_j,
    int dimension) {
    
    SparseResult result;
    
    #pragma omp parallel
    {
        // Thread-local storage
        vector<int> local_rows, local_cols;
        vector<int64_t> local_values;
        local_rows.reserve(1000);
        local_cols.reserve(1000);
        local_values.reserve(1000);
        
        // // Aggressively optimized dot product computation for CPU (cache, SIMD, OpenMP)
        // #pragma omp for schedule(dynamic, 4) collapse(1)
        // for (int i = 0; i < block_i.cols(); ++i) {
        //     const int* col_i = &block_i(0, i);

        //     // Prefetch next column of block_i to L1 cache (if available)
        //     if (i + 1 < block_i.cols()) {
        //         __builtin_prefetch(&block_i(0, i + 1), 0, 1);
        //     }

        //     for (int j = 0; j < block_j.cols(); ++j) {
        //         const int* col_j = &block_j(0, j);

        //         // Prefetch next column of block_j to L1 cache (if available)
        //         if (j + 1 < block_j.cols()) {
        //             __builtin_prefetch(&block_j(0, j + 1), 0, 1);
        //         }

        //         double threshold = 0.05 * (norms_i(i) + norms_j(j)); // norms are squared

        //         int64_t dot_product = 0;
        //         int k = 0;

        //         // SIMD-friendly loop: process in chunks of 8 (if possible)
        //         #if defined(__AVX2__)
        //             __m256i acc = _mm256_setzero_si256();
        //         for (; k <= dimension - 8; k += 8) {
        //             __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_i[k]));
        //             __m256i vj = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_j[k]));
        //             __m256i prod = _mm256_mullo_epi32(vi, vj);
        //             acc = _mm256_add_epi32(acc, prod);
        //         }
        //         // Horizontal sum of acc
        //         alignas(32) int32_t tmp[8];
        //         _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), acc);
        //         for (int t = 0; t < 8; ++t) dot_product += tmp[t];
        //         #endif

        //         // Fallback for remaining elements or non-AVX2
        //         for (; k <= dimension - 4; k += 4) {
        //             dot_product += static_cast<int64_t>(col_i[k]) * col_j[k];
        //             dot_product += static_cast<int64_t>(col_i[k+1]) * col_j[k+1];
        //             dot_product += static_cast<int64_t>(col_i[k+2]) * col_j[k+2];
        //             dot_product += static_cast<int64_t>(col_i[k+3]) * col_j[k+3];
        //         }
        //         for (; k < dimension; ++k) {
        //             dot_product += static_cast<int64_t>(col_i[k]) * col_j[k];
        //         }

        //         // Early exit if dot_product cannot reach threshold (optional, for sparse data)
        //         // Not implemented here due to integer overflow risk

        //         if (static_cast<double>(dot_product) / dimension > threshold) {
        //             local_rows.push_back(i);
        //             local_cols.push_back(j);
        //             local_values.push_back(dot_product);
        //         }
        //     }
        // }

        MatrixXi dot_products = block_i.transpose() * block_j;
        // Go through the solution and apply the threshold
        for (int i = 0; i < dot_products.rows(); ++i) {
            for (int j = 0; j < dot_products.cols(); ++j) {
                double threshold = 0.05 * (norms_i(i) + norms_j(j));
                int64_t dot_product = dot_products(i, j);
                if (dot_product / dimension > threshold) {
                    local_rows.push_back(i);
                    local_cols.push_back(j);
                    local_values.push_back(dot_product);
                }
            }
        }


        // Combine results from all threads
        #pragma omp critical
        {
            result.rows.insert(result.rows.end(), local_rows.begin(), local_rows.end());
            result.cols.insert(result.cols.end(), local_cols.begin(), local_cols.end());
            result.values.insert(result.values.end(), local_values.begin(), local_values.end());
        }
    }
    
    return result;
}

SparseResult compute_jaccard_with_MinHash(
    const MatrixXi& block_i, 
    const MatrixXi& block_j, 
    int dimension){

    SparseResult result;

    #pragma omp parallel
    {
        vector<int> local_rows, local_cols;
        vector<int64_t> local_values;
        local_rows.reserve(1000);
        local_cols.reserve(1000);
        local_values.reserve(1000);

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < block_i.cols(); ++i) {
            for (int j = 0; j < block_j.cols(); ++j) {
                const int* col_i = &block_i(0, i);
                const int* col_j = &block_j(0, j);

                int intersection = 0, union_count = 0;

                // Use simple loop, let compiler auto-vectorize

                int idx_i = 0, idx_j = 0;
                while (idx_i < dimension && idx_j < dimension) {
                    int vi = col_i[idx_i];
                    int vj = col_j[idx_j];
                    if (vi == 0 && vj == 0) {
                        ++idx_i;
                        ++idx_j;
                        continue;
                    }
                    if (vi == vj) {
                        if (vi != 0) {
                            ++intersection;
                            ++union_count;
                        }
                        ++idx_i;
                        ++idx_j;
                    } else if (vi == 0 || (vj != 0 && vi > vj)) {
                        ++union_count;
                        ++idx_j;
                    } else {
                        ++union_count;
                        ++idx_i;
                    }
                }
                // Count remaining non-zero elements in either vector
                while (idx_i < dimension) {
                    if (col_i[idx_i] != 0) ++union_count;
                    ++idx_i;
                }
                while (idx_j < dimension) {
                    if (col_j[idx_j] != 0) ++union_count;
                    ++idx_j;
                }

                if (union_count > 0) {
                    double jaccard = static_cast<double>(intersection) / union_count;
                    if (jaccard > 0.1) {
                        local_rows.push_back(i);
                        local_cols.push_back(j);
                        // Store jaccard as int64_t scaled by 1e6 for precision
                        local_values.push_back(static_cast<int64_t>(jaccard * 1e6));
                    }
                }
            }
        }

        #pragma omp critical
        {
            result.rows.insert(result.rows.end(), local_rows.begin(), local_rows.end());
            result.cols.insert(result.cols.end(), local_cols.begin(), local_cols.end());
            result.values.insert(result.values.end(), local_values.begin(), local_values.end());
        }
    }

    return result;
}

// Compute squared norms efficiently
VectorXd compute_norms_squared(const MatrixXi& matrix) {
    VectorXd norms(matrix.cols());
    
    #pragma omp parallel for
    for (int i = 0; i < matrix.cols(); ++i) {

        const int* row_i = &matrix(0, i);
        
        uint64_t norm_sq = 0;
        int k = 0;
        // Process in chunks of 4 for better vectorization
        for (; k <= matrix.rows() - 4; k += 4) {
            norm_sq += row_i[k] * row_i[k];
            norm_sq += row_i[k+1] * row_i[k+1];
            norm_sq += row_i[k+2] * row_i[k+2];
            norm_sq += row_i[k+3] * row_i[k+3];
        }
        
        // Handle remaining elements
        for (; k < matrix.rows(); ++k) {
            norm_sq += row_i[k] * row_i[k];
        }

        norms(i) = static_cast<double>(norm_sq);
    }
    
    return norms;
}

// Write sparse results to file (simple format)
void write_sparse_results(const string& folder, 
                         const vector<tuple<int, int, int64_t>>& results,
                         int dimension) {

    // Remove existing output folder if it exists, then create it
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }

    unordered_map<int, std::pair<vector<int>,vector<int64_t>>> reorganized_results;
    for (const auto& [row, col, value] : results) {
        reorganized_results[row].first.push_back(col);
        reorganized_results[row].second.push_back(value);
    }

    // Write binary output: int32, vector<int32>, vector<int32>(number_of_cols, vector:diff_of_cols_with_previous_col, vector:values/2048)
    string bin_filename = folder + "matrix.bin";
    ofstream bin_out(bin_filename, ios::binary);

    // File to store the position of the first byte for each row
    string index_filename = folder + "row_index.txt";
    ofstream index_out(index_filename);

    // Map from row to first byte position in the binary file
    int64_t current_pos = 0;

    // Write each row's results in the new format, iterating only over rows present in reorganized_results
    for (const auto& [row, pair] : reorganized_results) {
        const vector<int>& cols = pair.first;
        const vector<int64_t>& vals = pair.second;
        int32_t num_cols = static_cast<int32_t>(cols.size());

        // Record the first position for this row
        index_out << row << " " << current_pos << endl;

        // Write column indices as differences (deltas) from previous col
        int32_t prev_col = 0;
        for (size_t k = 0; k < cols.size(); ++k) {
            int32_t delta_col = cols[k] - prev_col;
            prev_col = cols[k];
            bin_out.write(reinterpret_cast<const char*>(&delta_col), sizeof(int32_t));
            current_pos += sizeof(int32_t);
        }

        // Write values (divided by 2048)
        for (size_t k = 0; k < vals.size(); ++k) {
            int32_t val32 = static_cast<int32_t>(round(static_cast<double>(vals[k]) / dimension));
            bin_out.write(reinterpret_cast<const char*>(&val32), sizeof(int32_t));
            current_pos += sizeof(int32_t);
        }
    }

    // Compress the output files using zstd and remove the originals
    string cmd1 = "zstd -f " + bin_filename + " && rm -f " + bin_filename;
    string cmd2 = "zstd -f " + index_filename + " && rm -f " + index_filename;
    system(cmd1.c_str());
    system(cmd2.c_str());
}

int main(int argc, char* argv[]) {
    // Argument parsing using clipp
    string matrix_file;
    int dimension = 0;
    double max_memory_gb = 0.0;
    int num_threads = 1;
    string output_folder;
    int num_shards = 1;
    int shard_idx = 0;
    int strategy = 0; // 0=random projections, 1=minHashes

    bool show_help = false;

    auto cli = (
        clipp::option("--vectors") & clipp::value("file", matrix_file),
        clipp::option("--dimension") & clipp::value("int", dimension),
        clipp::option("--max_memory_gb") & clipp::value("float", max_memory_gb),
        clipp::option("--num_threads") & clipp::value("int", num_threads),
        clipp::option("--output_folder") & clipp::value("folder", output_folder),
        clipp::option("--num_shards") & clipp::value("int", num_shards),
        clipp::option("--shard_idx") & clipp::value("int", shard_idx),
        clipp::option("--strategy") & clipp::value("int", strategy),
        clipp::option("--help").set(show_help)
    );

    if (!clipp::parse(argc, argv, cli) || show_help) {
        cout << "Usage:\n"
             << clipp::usage_lines(cli, argv[0]) << endl;
        cout << "\n--strategy 0=random projections (default), 1=minHashes\n";
        return show_help ? 0 : 1;
    }

    if (matrix_file.empty() || dimension <= 0 || max_memory_gb <= 0.0 || num_threads <= 0 || output_folder.empty() || num_shards <= 0 || shard_idx < 0 || shard_idx >= num_shards) {
        cerr << "Missing or invalid arguments. Use --help for usage." << endl;
        return 1;
    }

    string norms_file = output_folder + "vector_norms.txt";
    if (!fs::exists(norms_file)) {
        cerr << "Error: Required file 'vector_norms.txt' not found in output folder: " << output_folder << endl;
        return 1;
    }

    // for (const auto& entry : fs::directory_iterator(output_folder)) {
    //     if (entry.is_directory() && entry.path().filename().string().find("shard_") == 0) {
    //         fs::remove_all(entry.path());
    //     }
    // }

    omp_set_num_threads(num_threads);

    vector<double> all_norms;
    string line;
    ifstream norms_in(norms_file);
    while (getline(norms_in, line)) {
        size_t pos = line.find(' ');
        if (pos == string::npos) continue;
        double norm = stod(line.substr(pos + 1));
        all_norms.push_back(norm*norm);
    }

    // Ensure output folder ends with '/'
    if (!output_folder.empty() && output_folder.back() != '/' && output_folder.back() != '\\') {
        output_folder += '/';
    }

    // Output to subfolder for this shard
    string shard_folder = output_folder + "shard_" + to_string(shard_idx) + "/";
    if (!fs::exists(shard_folder)) {
        fs::create_directories(shard_folder);
    }

    // Calculate chunk size
    int bytes_per_vector = dimension * sizeof(int32_t);
    int64_t max_bytes = static_cast<int64_t>(max_memory_gb * 1024 * 1024 * 1024);
    cout << "max bytes " << max_bytes << " " << max_memory_gb << endl;
    int size_of_chunk = max_bytes / (bytes_per_vector * bytes_per_vector);

    cout << "Using chunks of size " << size_of_chunk << endl;

    // Get total number of vectors
    ifstream file(matrix_file, ios::ate | ios::binary);
    int64_t file_size = file.tellg();
    file.close();
    int total_vectors = file_size / bytes_per_vector;

    cout << "Total vectors: " << total_vectors << endl;

    // Compute row range for this shard
    int rows_per_shard = (total_vectors + num_shards - 1) / num_shards;
    int begin_row = shard_idx * rows_per_shard;
    int end_row = min(begin_row + rows_per_shard, total_vectors);

    cout << "Shard " << shard_idx << " processing rows " << begin_row << " to " << end_row << endl;

    vector<tuple<int, int, int64_t>> all_results;

    auto start_time = chrono::high_resolution_clock::now();

    for (int begin_i = begin_row; begin_i < end_row; begin_i += size_of_chunk) {
        int end_i = min(begin_i + size_of_chunk, end_row);
        MatrixXi block_i = load_matrix_block(matrix_file, dimension, begin_i, end_i);
        VectorXd norms_i = Map<VectorXd>(all_norms.data() + begin_i, end_i - begin_i);

        for (int begin_j = 0; begin_j < total_vectors; begin_j += size_of_chunk) {
            int end_j = min(begin_j + size_of_chunk, total_vectors);
            MatrixXi block_j = load_matrix_block(matrix_file, dimension, begin_j, end_j);
            VectorXd norms_j = Map<VectorXd>(all_norms.data() + begin_j, end_j - begin_j);

            cout << "Processing block (" << begin_i << ":" << end_i << ") x ("
                 << begin_j << ":" << end_j << ")" << endl;

            SparseResult result;
            if (strategy == 0){ //random projections
                result = compute_sparse_dot_products_optimized(block_i, block_j, norms_i, norms_j, dimension);
            }
            else{ //minHashes
                result = compute_jaccard_with_MinHash(block_i, block_j, dimension);
            }

            // Add global offsets and store
            for (size_t k = 0; k < result.values.size(); ++k) {
                all_results.emplace_back(
                    begin_i + result.rows[k],
                    begin_j + result.cols[k],
                    result.values[k]
                );
            }
        }
        cout << "qfklelmkqfsdj" << endl;
        break;
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Total computation time: " << duration.count() << " ms" << endl;
    cout << "Total results: " << all_results.size() << endl;

    // Write results to the shard subfolder
    write_sparse_results(shard_folder, all_results, dimension);
    
    return 0;
}