/*
 * Optimized pairwise comparison with int16 vectors
 * 
 * Key optimizations:
 * 1. Cache-line aligned memory layout (64-byte boundaries)
 * 2. Cache-aware block sizing (targeting L3 cache)
 * 3. AVX2 SIMD vectorization for int16 dot products
 * 4. Prefetching for reduced cache misses
 * 5. OpenMP parallelization with optimized scheduling
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <unordered_map>
#include <tuple>
#include <cstring>

#include "../bits/include/elias_fano.hpp"
#include "clipp.h"
    
namespace fs = std::filesystem;
using namespace std;

struct SparseResult {
    vector<int> rows;
    vector<int> cols;
    vector<int64_t> values;
};

// Simple matrix structure using contiguous memory
// Data is stored in column-major format: each vector is a column
// Rows are padded to 64-byte (cache line) boundaries for better alignment
struct Matrix16 {
    vector<int16_t> data;
    int rows;          // Actual number of rows
    int rows_padded;   // Padded to cache-line boundary
    int cols;
    
    Matrix16(int r = 0, int c = 0) : rows(r), cols(c) {
        // Pad rows to 32 elements (64 bytes for int16_t) for cache-line alignment
        rows_padded = ((r + 31) / 32) * 32;
        data.resize(rows_padded * c, 0);  // Initialize padding with zeros
    }
    
    int16_t* get_col(int col_idx) {
        return data.data() + col_idx * rows_padded;
    }
    
    const int16_t* get_col(int col_idx) const {
        return data.data() + col_idx * rows_padded;
    }
};

struct Matrix32 {
    vector<int32_t> data;
    int rows;          // Actual number of rows
    int rows_padded;   // Padded to cache-line boundary
    int cols;
    
    Matrix32(int r = 0, int c = 0) : rows(r), cols(c) {
        // Pad rows to 16 elements (64 bytes for int32_t) for cache-line alignment
        rows_padded = ((r + 15) / 16) * 16;
        data.resize(rows_padded * c, 0);  // Initialize padding with zeros
    }
    
    int32_t* get_col(int col_idx) {
        return data.data() + col_idx * rows_padded;
    }
    
    const int32_t* get_col(int col_idx) const {
        return data.data() + col_idx * rows_padded;
    }
};

// Load a block of vectors from binary file as int16
Matrix16 load_matrix_block_int16(const string& file_path, int dimension, int begin, int end) {
    ifstream file(file_path, ios::binary);
    if (!file) {
        cerr << "Error opening file: " << file_path << endl;
        return Matrix16();
    }
    
    uint64_t vector_size = dimension * sizeof(int16_t);
    file.seekg(begin * vector_size);
    int num_vectors = end - begin;
    
    Matrix16 matrix(dimension, num_vectors);
    
    // Read vectors one by one into padded layout
    vector<int16_t> temp_buffer(dimension);
    for (int i = 0; i < num_vectors; ++i) {
        file.read(reinterpret_cast<char*>(temp_buffer.data()), vector_size);
        int16_t* col = matrix.get_col(i);
        memcpy(col, temp_buffer.data(), vector_size);
        // Padding bytes are already zero-initialized
    }
    
    if (!file) {
        cerr << "Error reading file: " << file_path << endl;
        return Matrix16();
    }
    
    return matrix;
}

// Load a block of vectors from binary file as int32 (for MinHash)
Matrix32 load_matrix_block_int32(const string& file_path, int dimension, int begin, int end) {
    ifstream file(file_path, ios::binary);
    if (!file) {
        cerr << "Error opening file: " << file_path << endl;
        return Matrix32();
    }
    
    uint64_t vector_size = dimension * sizeof(int16_t);
    file.seekg(begin * vector_size);
    int num_vectors = end - begin;
    
    Matrix32 matrix(dimension, num_vectors);
    
    // Read and convert vectors one by one into padded layout
    vector<int16_t> temp_buffer(dimension);
    for (int i = 0; i < num_vectors; ++i) {
        file.read(reinterpret_cast<char*>(temp_buffer.data()), vector_size);
        int32_t* col = matrix.get_col(i);
        for (int j = 0; j < dimension; ++j) {
            col[j] = static_cast<int32_t>(temp_buffer[j]);
        }
        // Padding elements are already zero-initialized
    }
    
    if (!file) {
        cerr << "Error reading file: " << file_path << endl;
        return Matrix32();
    }
    
    return matrix;
}

// Optimized sparse dot product computation with early threshold checking
SparseResult compute_sparse_dot_products_optimized(
    const Matrix16& block_i, 
    const Matrix16& block_j, 
    const vector<double>& norms_i, 
    const vector<double>& norms_j,
    int dimension) {
    
    SparseResult result;
    result.rows.reserve(10000);
    result.cols.reserve(10000);
    result.values.reserve(10000);
    
    #pragma omp parallel
    {
        // Thread-local storage with large reservation to reduce reallocation
        vector<int> local_rows, local_cols;
        vector<int64_t> local_values;
        local_rows.reserve(5000);
        local_cols.reserve(5000);
        local_values.reserve(5000);
        
        #pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < block_i.cols; ++i) {
            const int16_t* col_i = block_i.get_col(i);

            // Prefetch next column to L1 cache
            if (i + 1 < block_i.cols) {
                __builtin_prefetch(block_i.get_col(i + 1), 0, 1);
            }

            for (int j = 0; j < block_j.cols; ++j) {
                // auto dot_start = chrono::high_resolution_clock::now();
                const int16_t* col_j = block_j.get_col(j);

                // Prefetch next column to L1 cache
                if (j + 1 < block_j.cols) {
                    __builtin_prefetch(block_j.get_col(j + 1), 0, 1);
                }

                int32_t dot_product = 0;
                int k = 0;

                #if defined(__AVX2__)
                    // AVX2 SIMD optimization for int16 dot product - optimized for 2048 elements
                    __m256i acc1 = _mm256_setzero_si256();
                    __m256i acc2 = _mm256_setzero_si256();
                    
                    // Unroll loop to process 32 elements (2x16) per iteration
                    for (; k <= dimension - 32; k += 32) {
                        // Prefetch data 2 iterations ahead (64 elements = 128 bytes)
                        if (k + 64 < dimension) {
                            __builtin_prefetch(&col_i[k + 64], 0, 1);
                            __builtin_prefetch(&col_j[k + 64], 0, 1);
                        }
                        
                        // Load and process first 16 elements
                        __m256i vi1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_i[k]));
                        __m256i vj1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_j[k]));
                        __m256i prod1 = _mm256_madd_epi16(vi1, vj1);
                        acc1 = _mm256_add_epi32(acc1, prod1);
                        
                        // Load and process next 16 elements
                        __m256i vi2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_i[k + 16]));
                        __m256i vj2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_j[k + 16]));
                        __m256i prod2 = _mm256_madd_epi16(vi2, vj2);
                        acc2 = _mm256_add_epi32(acc2, prod2);
                    }
                    
                    // Handle remaining 16 elements if dimension isn't multiple of 32
                    for (; k <= dimension - 16; k += 16) {
                        __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_i[k]));
                        __m256i vj = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&col_j[k]));
                        __m256i prod = _mm256_madd_epi16(vi, vj);
                        acc1 = _mm256_add_epi32(acc1, prod);
                    }
                    
                    // Combine both accumulators and reduce to single value
                    __m256i final_acc = _mm256_add_epi32(acc1, acc2);
                    __m128i acc_low = _mm256_extracti128_si256(final_acc, 0);
                    __m128i acc_high = _mm256_extracti128_si256(final_acc, 1);
                    __m128i sum128 = _mm_add_epi32(acc_low, acc_high);
                    sum128 = _mm_hadd_epi32(sum128, sum128);
                    sum128 = _mm_hadd_epi32(sum128, sum128);
                    dot_product += _mm_extract_epi32(sum128, 0);
                #elif defined(__SSE2__)
                    // SSE2 fallback for int16 dot product
                    __m128i acc = _mm_setzero_si128();
                    for (; k <= dimension - 8; k += 8) {
                        __m128i vi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col_i[k]));
                        __m128i vj = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col_j[k]));
                        __m128i prod = _mm_madd_epi16(vi, vj);
                        acc = _mm_add_epi32(acc, prod);
                    }
                    
                    // Extract and sum the 4 int32 values
                    acc = _mm_hadd_epi32(acc, acc);
                    acc = _mm_hadd_epi32(acc, acc);
                    dot_product += _mm_extract_epi32(acc, 0);
                #else
                    // Process in chunks of 4 for better vectorization
                    for (; k <= dimension - 4; k += 4) {
                        dot_product += static_cast<int32_t>(col_i[k]) * col_j[k];
                        dot_product += static_cast<int32_t>(col_i[k+1]) * col_j[k+1];
                        dot_product += static_cast<int32_t>(col_i[k+2]) * col_j[k+2];
                        dot_product += static_cast<int32_t>(col_i[k+3]) * col_j[k+3];
                    }
                #endif
                                
                    // Handle remaining elements
                    for (; k < dimension; ++k) {
                        dot_product += static_cast<int32_t>(col_i[k]) * col_j[k];
                    }


                auto thresh_start = chrono::high_resolution_clock::now();
                double threshold = 0.05 * (norms_i[i] + norms_j[j]);

                // if (i==2 && j==274){
                //     cout << "qfidosn " << dot_product << " " << norms_i[i] << " " << norms_j[j] << " " << static_cast<double>(dot_product) / dimension << " > " << threshold << endl;
                //     // exit(1);
                // }

                if (static_cast<double>(dot_product) / dimension > threshold) {
                    local_rows.push_back(i);
                    local_cols.push_back(j);
                    local_values.push_back(dot_product);
                }

                // auto dot_end = chrono::high_resolution_clock::now();
                // auto thresh_end = chrono::high_resolution_clock::now();

                // auto dot_duration = chrono::duration_cast<chrono::nanoseconds>(thresh_start - dot_start);
                // auto thresh_duration = chrono::duration_cast<chrono::nanoseconds>(thresh_end - thresh_start);
                
                // cout << "Dot product time: " << dot_duration.count() << " ns, Threshold check time: " << thresh_duration.count() << " ns" << endl;
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
    const Matrix32& block_i, 
    const Matrix32& block_j, 
    int dimension) {

    SparseResult result;

    #pragma omp parallel
    {
        vector<int> local_rows, local_cols;
        vector<int64_t> local_values;
        local_rows.reserve(1000);
        local_cols.reserve(1000);
        local_values.reserve(1000);

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < block_i.cols; ++i) {
            for (int j = 0; j < block_j.cols; ++j) {
                const int32_t* col_i = block_i.get_col(i);
                const int32_t* col_j = block_j.get_col(j);

                int intersection = 0, union_count = 0;

                int idx_i = 0, idx_j = 0;
                while (idx_i < dimension && idx_j < dimension) {
                    int32_t vi = col_i[idx_i];
                    int32_t vj = col_j[idx_j];
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

// Compute squared norms efficiently for int16 matrix
vector<double> compute_norms_squared_int16(const Matrix16& matrix) {
    vector<double> norms(matrix.cols);
    
    #pragma omp parallel for
    for (int i = 0; i < matrix.cols; ++i) {
        const int16_t* col = matrix.get_col(i);
        
        uint64_t norm_sq = 0;
        int k = 0;
        
        // Process in chunks of 4 for better vectorization
        for (; k <= matrix.rows - 4; k += 4) {
            norm_sq += static_cast<uint64_t>(col[k]) * col[k];
            norm_sq += static_cast<uint64_t>(col[k+1]) * col[k+1];
            norm_sq += static_cast<uint64_t>(col[k+2]) * col[k+2];
            norm_sq += static_cast<uint64_t>(col[k+3]) * col[k+3];
        }
        
        // Handle remaining elements
        for (; k < matrix.rows; ++k) {
            norm_sq += static_cast<uint64_t>(col[k]) * col[k];
        }

        norms[i] = static_cast<double>(norm_sq);
    }
    
    return norms;
}

// Write sparse results to file (simple format)
void write_sparse_results_prev(const string& folder, 
                         const vector<tuple<int, int, int64_t>>& results,
                         int dimension) {

    // Create output folder if it doesn't exist
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }

    unordered_map<int, pair<vector<int32_t>, vector<int64_t>>> reorganized_results;
    for (const auto& [row, col, value] : results) {
        reorganized_results[row].first.push_back(col);
        reorganized_results[row].second.push_back(value);
    }

    string bin_filename = folder + "matrix.bin";
    ofstream bin_out(bin_filename, ios::binary);

    string index_filename = folder + "row_index.txt";
    ofstream index_out(index_filename);

    int64_t current_pos = 0;

    // Write each row's results in the new format
    for (const auto& [row, pair] : reorganized_results) {
        const vector<int32_t>& cols = pair.first;
        const vector<int64_t>& vals = pair.second;

        // Record the first position for this row
        index_out << row << " " << current_pos << endl;

        // Write column indices as differences (deltas) from previous col
        int32_t prev_col = cols[0];
        bin_out.write(reinterpret_cast<const char*>(&prev_col), sizeof(int32_t));
        current_pos += sizeof(int32_t);
        
        for (size_t k = 1; k < cols.size(); ++k) {
            int32_t delta_col = cols[k] - prev_col;
            prev_col = cols[k];
            bin_out.write(reinterpret_cast<const char*>(&delta_col), sizeof(int32_t));
            current_pos += sizeof(int32_t);
        }
        
        // Write values (divided by dimension)
        for (size_t k = 0; k < vals.size(); ++k) {
            int32_t val32 = static_cast<int32_t>(round(static_cast<double>(vals[k]) / dimension));
            bin_out.write(reinterpret_cast<const char*>(&val32), sizeof(int32_t));
            current_pos += sizeof(int32_t);
        }
    }

    bin_out.close();
    index_out.close();

    // Compress the output files using zstd and remove the originals
    string cmd1 = "zstd -f " + bin_filename + " && rm -f " + bin_filename;
    string cmd2 = "zstd -f " + index_filename + " && rm -f " + index_filename;
    system(cmd1.c_str());
    system(cmd2.c_str());
}

void write_sparse_results(const string& folder, 
                         const vector<tuple<int, int, int64_t>>& results,
                         int dimension) {

    // Create output folder if it doesn't exist
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }

    unordered_map<int, pair<vector<int32_t>, vector<int64_t>>> reorganized_results;
    for (const auto& [row, col, value] : results) {
        reorganized_results[row].first.push_back(col);
        reorganized_results[row].second.push_back(value);
    }

    string bin_filename = folder + "matrix.bin";
    ofstream bin_out(bin_filename, ios::binary);

    string index_filename = folder + "row_index.bin";
    ofstream index_out(index_filename, ios::binary);

    int64_t current_pos = 0;

    auto get_dot_products_vec = [&](const vector<int64_t>& vals) {
        vector<int64_t> dot_products_vec(vals.size());
        for (size_t k = 0; k < vals.size(); ++k) {
            dot_products_vec[k] = static_cast<int64_t>(round(static_cast<double>(vals[k]) / dimension));
        }
        return dot_products_vec;
    };

    vector<int32_t> row_vec(reorganized_results.size());
    vector<int64_t> curr_pos_vec(reorganized_results.size());

    // Write each row's results in the new format
    int indx = 0;
    for (const auto& [row, pair] : reorganized_results) {
        const vector<int32_t>& cols = pair.first;
        const vector<int64_t>& vals = pair.second;

        row_vec[indx] = row;
        curr_pos_vec[indx++] = current_pos;
        
        bits::elias_fano<> ef;
        ef.encode(cols.begin(), cols.size(), cols.back() + 1);
        ef.save(bin_out);
        current_pos += ef.num_bytes();

        vector<int64_t> dot_products_vec = get_dot_products_vec(vals);
        bits::compact_vector cv;
        cv.build(dot_products_vec.begin(), dot_products_vec.size());
        cv.save(bin_out);
        current_pos += cv.num_bytes();
    }
    
    bin_out.close();
    
    bits::compact_vector cv_rows;
    cv_rows.build(row_vec.begin(), row_vec.size());
    cv_rows.save(index_out);
    
    bits::compact_vector cv_pos;
    cv_pos.build(curr_pos_vec.begin(), curr_pos_vec.size());
    cv_pos.save(index_out);
    
    index_out.close();

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
    int num_threads = 1;
    string output_folder;
    int num_shards = 1;
    int shard_idx = 0;
    int strategy = 0; // 0=random projections, 1=minHashes

    bool show_help = false;

    auto cli = (
        clipp::option("--vectors") & clipp::value("file", matrix_file),
        clipp::option("--dimension") & clipp::value("int", dimension),
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

    if (matrix_file.empty() || dimension <= 0 || num_threads <= 0 || 
        output_folder.empty() || num_shards <= 0 || shard_idx < 0 || shard_idx >= num_shards) {
        cerr << "Missing or invalid arguments. Use --help for usage." << endl;
        return 1;
    }

    // Ensure output folder ends with '/'
    if (!output_folder.empty() && output_folder.back() != '/' && output_folder.back() != '\\') {
        output_folder += '/';
    }

    string norms_file = output_folder + "vector_norms.txt";
    if (!fs::exists(norms_file)) {
        cerr << "Error: Required file 'vector_norms.txt' not found in output folder: " << output_folder << endl;
        return 1;
    }

    omp_set_num_threads(num_threads);

    // Load norms
    vector<double> all_norms;
    string line;
    ifstream norms_in(norms_file);
    while (getline(norms_in, line)) {
        size_t pos = line.find(' ');
        if (pos == string::npos) continue;
        double norm = stod(line.substr(pos + 1));
        all_norms.push_back(norm * norm); // Store squared norms
    }
    norms_in.close();

    // Output to subfolder for this shard
    string shard_folder = output_folder + "shard_" + to_string(shard_idx) + "/";
    if (!fs::exists(shard_folder)) {
        fs::create_directories(shard_folder);
    }

    // L3 cache size heuristic: aim to keep one block under 16MB for better cache utilization
    // Each block uses: dimension * sizeof(int16_t) * num_vectors bytes
    // For two blocks (block_i and block_j): 2 * dimension * sizeof(int16_t) * chunk_size
    int bytes_per_vector = dimension * sizeof(int16_t);
    int64_t target_cache_size = 16 * 1024 * 1024;  // 16 MB
    int size_of_chunk = target_cache_size / (2 * bytes_per_vector);
    
    // Ensure minimum chunk size for reasonable parallelism
    size_of_chunk = max(size_of_chunk, 64);
    
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

        for (int begin_j = 0; begin_j < total_vectors; begin_j += size_of_chunk) {
            int end_j = min(begin_j + size_of_chunk, total_vectors);

            cout << "Processing block (" << begin_i << ":" << end_i << ") x ("
                << begin_j << ":" << end_j << ")" << endl;

            SparseResult result;
            
            if (strategy == 0) { // random projections
                Matrix16 block_i = load_matrix_block_int16(matrix_file, dimension, begin_i, end_i);
                Matrix16 block_j = load_matrix_block_int16(matrix_file, dimension, begin_j, end_j);
                
                // Extract norms for this block
                vector<double> norms_i(all_norms.begin() + begin_i, all_norms.begin() + end_i);
                vector<double> norms_j(all_norms.begin() + begin_j, all_norms.begin() + end_j);
                
                result = compute_sparse_dot_products_optimized(block_i, block_j, norms_i, norms_j, dimension);
            }
            else { // minHashes
                Matrix32 block_i = load_matrix_block_int32(matrix_file, dimension, begin_i, end_i);
                Matrix32 block_j = load_matrix_block_int32(matrix_file, dimension, begin_j, end_j);
                
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
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Total computation time: " << duration.count() << " ms" << endl;
    cout << "Total results: " << all_results.size() << endl;

    

    // Write results to the shard subfolder
    write_sparse_results_prev(shard_folder, all_results, dimension);
    // write_sparse_results(shard_folder, all_results, dimension);
    
    return 0;
}
