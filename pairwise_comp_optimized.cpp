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
#include <immintrin.h>
    
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

        // Instead of manual dot product, use Eigen for block multiplication
        // Compute the dot product matrix
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
    // Argument parsing using -- syntax
    string matrix_file;
    int dimension = 0;
    double max_memory_gb = 0.0;
    int num_threads = 1;
    string output_folder;

    int num_shards = 1;
    int shard_idx = 0;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--vectors" && i + 1 < argc) {
            matrix_file = argv[++i];
        } else if (arg == "--dimension" && i + 1 < argc) {
            dimension = stoi(argv[++i]);
        } else if (arg == "--max_memory_gb" && i + 1 < argc) {
            max_memory_gb = stod(argv[++i]);
        } else if (arg == "--num_threads" && i + 1 < argc) {
            num_threads = stoi(argv[++i]);
        } else if (arg == "--output_folder" && i + 1 < argc) {
            output_folder = argv[++i];
        } else if (arg == "--num_shards" && i + 1 < argc) {
            num_shards = stoi(argv[++i]);
        } else if (arg == "--shard_idx" && i + 1 < argc) {
            shard_idx = stoi(argv[++i]);
        } else if (arg == "--help") {
            cout << "Usage: " << argv[0]
                 << " --vectors <file> --dimension <int> --max_memory_gb <float> --num_threads <int> --output_folder <folder> --num_shards <int> --shard_idx <int>" << endl;
            return 0;
        }
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

        auto t_blocki_start = chrono::high_resolution_clock::now();
        MatrixXi block_i = load_matrix_block(matrix_file, dimension, begin_i, end_i);
        VectorXd norms_i = Map<VectorXd>(all_norms.data() + begin_i, end_i - begin_i);

        auto t_blocki_end = chrono::high_resolution_clock::now();

        for (int begin_j = 0; begin_j < total_vectors; begin_j += size_of_chunk) {
            int end_j = min(begin_j + size_of_chunk, total_vectors);

            auto t_blockj_start = chrono::high_resolution_clock::now();
            MatrixXi block_j = load_matrix_block(matrix_file, dimension, begin_j, end_j);
            VectorXd norms_j = Map<VectorXd>(all_norms.data() + begin_j, end_j - begin_j);
            auto t_blockj_end = chrono::high_resolution_clock::now();

            cout << "Processing block (" << begin_i << ":" << end_i << ") x ("
                << begin_j << ":" << end_j << ")" << endl;

            auto t_dot_start = chrono::high_resolution_clock::now();
            SparseResult result;
            result = compute_sparse_dot_products_optimized(block_i, block_j, norms_i, norms_j, dimension);
            auto t_dot_end = chrono::high_resolution_clock::now();

            auto t_store_start = chrono::high_resolution_clock::now();
            // Add global offsets and store
            for (size_t k = 0; k < result.values.size(); ++k) {
                all_results.emplace_back(
                    begin_i + result.rows[k],
                    begin_j + result.cols[k],
                    result.values[k]
                );
            }
            auto t_store_end = chrono::high_resolution_clock::now();

            auto blocki_ms = chrono::duration_cast<chrono::milliseconds>(t_blocki_end - t_blocki_start).count();
            auto blockj_ms = chrono::duration_cast<chrono::milliseconds>(t_blockj_end - t_blockj_start).count();
            auto dot_ms = chrono::duration_cast<chrono::milliseconds>(t_dot_end - t_dot_start).count();
            auto store_ms = chrono::duration_cast<chrono::milliseconds>(t_store_end - t_store_start).count();

            cout << "  Timing: block_i=" << blocki_ms << "ms, block_j=" << blockj_ms
                 << "ms, dot=" << dot_ms << "ms, store=" << store_ms << "ms. Outputted " << result.rows.size() << " vecotrs " << endl;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Total computation time: " << duration.count() << " ms" << endl;
    cout << "Total results: " << all_results.size() << endl;

    // Write results to the shard subfolder
    write_sparse_results(shard_folder, all_results, dimension);
    
    return 0;
}
