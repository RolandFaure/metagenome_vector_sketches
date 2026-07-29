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

//#include <immintrin.h>
// Forward declaration instead of including .cpp file
int pairwise_comp_optimized_16bits(std::string db_folder, int num_threads, std::string output_folder, int dimension, int num_shards,int shard_idx);
#include "elias_fano.hpp"
#include "clipp.h"
#include "rice_sequence.hpp"
// #include "streamvbyte.h"
// #include "streamvbytedelta.h"

    
namespace fs = std::filesystem;
using namespace Eigen;
using namespace std;

struct SparseResult {
    vector<int> rows;
    vector<int> cols;
    vector<int64_t> values;
};

using MatrixXll = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

// Load a block of vectors from binary file
MatrixXll load_matrix_block(const string& file_path, int dimension, int begin, int end) {
    ifstream file(file_path, ios::binary);
    if (!file) {
        cerr << "Error opening file: " << file_path << endl;
        return MatrixXll();
    }

    uint64_t vector_size = dimension * sizeof(int16_t);
    file.seekg(begin * vector_size);
    int num_vectors = end - begin;
    vector<int16_t> buffer(num_vectors * dimension);
    file.read(reinterpret_cast<char*>(buffer.data()), num_vectors * vector_size);
    
    MatrixXll matrix(dimension, num_vectors);
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < dimension; ++j) {
            matrix(j, i) = buffer[i * dimension + j];
        }
    }
    
    return matrix;
}

void compare_matrix(MatrixXll mat1, MatrixXll mat2, int shard){
    for(size_t i=0; i<mat1.rows(); i++){
        for(size_t j=0; j<mat1.cols(); j++){
            if(mat1(i,j) != mat2(i,j)){
                std::cout<<"mismatch "<<shard<<" "<<i<<" "<<j<<std::endl;
                std::cout<<mat1(i,j)<<" "<<mat2(i,j)<<std::endl;
                exit(1);
            }
        }

    }
}

// Optimized sparse dot product computation with early threshold checking
SparseResult compute_sparse_dot_products_optimized(
    const MatrixXll& block_i, 
    const MatrixXll& block_j, 
    const VectorXd& norms_i, 
    const VectorXd& norms_j,
    int dimension) {
    
    SparseResult result;
    
    vector<int> local_rows, local_cols;
    vector<int64_t> local_values;
    local_rows.reserve(1000);
    local_cols.reserve(1000);
    local_values.reserve(1000);
    
    MatrixXll dot_products = block_i.transpose() * block_j;
    // Go through the solution and apply the threshold
    for (int i = 0; i < dot_products.rows(); ++i) {
        for (int j = 0; j < dot_products.cols(); ++j) {
            double threshold = 0.05 * (norms_i(i) + norms_j(j));
            int64_t dot_product = dot_products(i, j);
            if (static_cast<double>(dot_product) / dimension > threshold) { 
                local_rows.push_back(i);
                local_cols.push_back(j);
                local_values.push_back(dot_product);
            }
        }
    }
    result.rows.insert(result.rows.end(), local_rows.begin(), local_rows.end());
    result.cols.insert(result.cols.end(), local_cols.begin(), local_cols.end());
    result.values.insert(result.values.end(), local_values.begin(), local_values.end());
    return result;
}

void write_matrix(const string& folder, 
                         const vector<tuple<int, int, int64_t>>& results,
                         const vector<double>& all_norms_vec,
                         int dimension,
                        uint64_t begin_row, uint64_t end_row) {

    // Remove existing output folder if it exists, then create it
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }
    const double MULT_CONST = (1ULL << 8) - 1;
    // Contains for each accession, its neighbor and corresponding Jaccard
    unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint16_t> > >reorganized_results;
    for (const auto& [row, col, value] : results) {
        double norm_curr = all_norms_vec[row];
        double norm_col = all_norms_vec[col];
        double inter_col = static_cast<double>(value)/dimension;
        double jaccard = inter_col / (norm_curr + norm_col - inter_col);
        if(jaccard > 1) jaccard = 1;
        uint16_t quantized_jaccard = static_cast<uint16_t>(round(jaccard * MULT_CONST));
        reorganized_results[row].push_back(std::make_pair(col, quantized_jaccard));
    }

    // Write binary output: int32, vector<int32>, vector<int32>(number_of_cols, vector:diff_of_cols_with_previous_col, vector:values/2048)
    string bin_filename = folder + "/matrix.bin";
    ofstream bin_out(bin_filename, ios::binary);

    // File to store the position of the first byte for each row (offset array)
    string index_filename = folder + "/row_index.bin";
    ofstream index_out(index_filename, ios::binary);

    // std::vector<uint32_t> row_vec(reorganized_results.size());
    std::vector<uint64_t> curr_pos_vec(reorganized_results.size());
    std::vector<uint32_t> start_neighbor(reorganized_results.size());

    
    // std::ofstream temp_out("space_usage.txt");
    // uint64_t jac_space = 0, ngh_space = 0;
    // Write each row's results in the new format, iterating only over rows present in reorganized_results
    int indx = 0;
    // neighbor_pair_vec<neighbor_index, jaccard_btwn_me_&_neighbor>
    // for (auto& [row, neighbor_pair_vec] : reorganized_results) {
    for (uint64_t row=begin_row; row<end_row; row++){
        auto neighbor_pair_vec = reorganized_results[row];
        // sort(neighbor_pair_vec.begin(), neighbor_pair_vec.end(),[] (const std::pair<uint32_t, uint16_t>& a, const std::pair<uint32_t, uint16_t>& b) {
        //     return a.second > b.second;
        // });

        std::vector<uint32_t> neighbor_indx_vec;
        std::vector<uint16_t> neighbor_jaccard_vec;
        neighbor_indx_vec.reserve(neighbor_pair_vec.size());
        neighbor_jaccard_vec.reserve(neighbor_pair_vec.size());

        for (auto& [idx, j] : neighbor_pair_vec) {
            neighbor_indx_vec.push_back(idx);
            neighbor_jaccard_vec.push_back(j);
        }

        // Record this row index and its stored position 
        uint64_t current_pos = static_cast<uint64_t>(bin_out.tellp());
        
        // FIXME: The row values should be just consecutive values [everyone has at least one neighbor (self)], hence do not need to store these
        // the first row value can be calculated from the shard value

        // row_vec[indx] = row; 
        // std::cout<<"Row: "<<row<<std::endl;

        curr_pos_vec[indx] = current_pos;
        start_neighbor[indx++] = neighbor_indx_vec[0];
        
        std::vector<uint64_t> delta_cols(neighbor_indx_vec.size()-1);
        for (size_t k = 1; k < neighbor_indx_vec.size(); ++k) {
            assert(neighbor_indx_vec[k] > neighbor_indx_vec[k-1]);
            delta_cols[k-1] = neighbor_indx_vec[k] - neighbor_indx_vec[k-1];
        }

        bits::compact_vector cv_jc;
        cv_jc.build(neighbor_jaccard_vec.begin(), neighbor_jaccard_vec.size());
        cv_jc.save(bin_out);
        
        assert(neighbor_jaccard_vec.size() >= 1);
        
        if(neighbor_jaccard_vec.size() == 1) continue;
        
        bits::rice_sequence<> rs_delta;
        rs_delta.encode(delta_cols.begin(), delta_cols.size());
        rs_delta.save(bin_out);
    }
    bin_out.flush();     
    bin_out.close();
    
    // bits::compact_vector cv_rows;
    // cv_rows.build(row_vec.begin(), row_vec.size());
    // cv_rows.save(index_out);

    // curr_pos_vec is sorted
    std::vector<uint64_t> curr_pos_delta_vec(curr_pos_vec.size()-1);
    // curr_pos_vec[0] is always 0;
    for(size_t i=1; i<curr_pos_vec.size(); i++){
        curr_pos_delta_vec[i-1] = curr_pos_vec[i] - curr_pos_vec[i-1];
    }

    bits::compact_vector cv_cps; // Compact Vector Current PositionS
    cv_cps.build(curr_pos_delta_vec.begin(), curr_pos_delta_vec.size());
    cv_cps.save(index_out);
    index_out.close();

    std::string neighbor_fn = folder + "neighbor_start.bin";
    std::ofstream ngh_out(neighbor_fn, std::ios::binary);
    bits::rice_sequence<> rs_start; 
    rs_start.encode(start_neighbor.begin(), start_neighbor.size());
    rs_start.save(ngh_out);
    ngh_out.close();
}


int main(int argc, char* argv[]) {
    // Argument parsing using clipp
    string db_folder, matrix_file;
    int dimension = 2048;
    double max_memory_gb = 1;
    int num_threads = 1;
    string output_folder;
    int num_shards = 1;
    
    // **NOTE: Change this whenever the file encoding is modified**
    const int encoding_version = 1;

    bool show_help = false;

    auto cli = (
        clipp::required("--db") & clipp::value("folder", db_folder),
        clipp::required("--output_folder") & clipp::value("folder", output_folder),
        clipp::option("--num_shards") & clipp::value("int", num_shards),
        clipp::option("--max_memory_gb") & clipp::value("float", max_memory_gb),
        clipp::option("--num_threads") & clipp::value("int", num_threads),
        clipp::option("--help").set(show_help)
    );

    if (!clipp::parse(argc, argv, cli) || show_help) {
        cout << "Create Pairwise Comparison Matrix\n\n";
        cout << "Usage:\n" << clipp::usage_lines(cli, argv[0]) << "\n\n";
        cout << "Options:\n";   
        cout << "  --db              Folder containing the matrix meta data [Required]\n";
        cout << "  --output_folder   Folder where to store the matrix [Required]\n";
        cout << "  --num_threads     Numer of threads to use [default 1]\n";
        cout << "  --num_shards      Number of shards to use [default 1]\n";
        cout << "  --max_memory_gb   Max memory to be used per shard [default 1 GB]\n";
        cout << "  --help            Show this help message\n\n";
        return show_help ? 0 : 1;
    }
    
    if (!output_folder.empty() && output_folder.back() != '/' && output_folder.back() != '\\') {
        output_folder += '/';
    }

    if (!db_folder.empty() && db_folder.back() != '/' && db_folder.back() != '\\') {
        db_folder += '/';
    }

    string norms_file = db_folder + "/vector_norms.txt";
    if (!fs::exists(norms_file)) {
        cerr << "Error: Required file 'vector_norms.txt' not found in output folder: " << db_folder << endl;
        return 1;
    }

    string dimension_file = db_folder + "/dimension.txt";
    if (fs::exists(dimension_file)) {
        ifstream dim_in(dimension_file);
        if (dim_in) {
            dim_in >> dimension;
            dim_in.close();
        }
    }

    string version_file = db_folder + "/version.txt";
    ofstream enc_out(version_file);
    enc_out<<encoding_version;
    enc_out.close();
    

    matrix_file = db_folder + "vectors.bin";

    vector<double> all_norms;
    string line;
    ifstream norms_in(norms_file);
    while (getline(norms_in, line)) {
        size_t pos = line.find(' ');
        if (pos == string::npos) continue;
        double norm = stod(line.substr(pos + 1));
        all_norms.push_back(norm*norm);
    }
    // Calculate chunk size
    int bytes_per_vector = dimension * sizeof(int16_t);
    int64_t max_bytes = static_cast<int64_t>(max_memory_gb * 1024 * 1024 * 1024);
    // cout << "max bytes " << max_bytes << " " << max_memory_gb << endl;
    int size_of_chunk = max_bytes / (bytes_per_vector * bytes_per_vector);

    cout << "Using chunks of size " << size_of_chunk <<" dimension: "<<dimension << endl;

    // Get total number of vectors
    ifstream file(matrix_file, ios::ate | ios::binary);
    int64_t file_size = file.tellg();
    file.close();
    int total_vectors = file_size / bytes_per_vector;

    cout << "Total vectors: " << total_vectors << endl;

    auto start_time = chrono::high_resolution_clock::now();
    
    // int outer_threads = min(num_threads, min(8, num_shards));
    // int inner_threads = max(1, num_threads / outer_threads);
    // omp_set_nested(1);
    // omp_set_max_active_levels(2);

    // omp_set_num_threads(outer_threads);
    // #pragma omp parallel for schedule(dynamic)
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)

    for(size_t shard_idx=0; shard_idx < num_shards; shard_idx++){
        string shard_folder = output_folder + "shard_" + to_string(shard_idx) + "/";
        if (!fs::exists(shard_folder)) {
            fs::create_directories(shard_folder);
        }

        // Compute row range for this shard
        int rows_per_shard = (total_vectors + num_shards - 1) / num_shards;
        int begin_row = shard_idx * rows_per_shard;
        int end_row = min(begin_row + rows_per_shard, total_vectors);

        // begin_row = 1372094;
        // end_row = begin_row + 2;
        #pragma omp critical
        {
            cout << "Shard " << shard_idx << " processing rows " << begin_row << " to " << end_row-1 << endl;
        }
        if(begin_row >= end_row) continue;

        vector<tuple<int, int, int64_t>> all_results;

        for (int begin_i = begin_row; begin_i < end_row; begin_i += size_of_chunk) {
            int end_i = min(begin_i + size_of_chunk, end_row);

            // auto t_blocki_start = chrono::high_resolution_clock::now();
            MatrixXll block_i = load_matrix_block(matrix_file, dimension, begin_i, end_i);
            VectorXd norms_i = Map<VectorXd>(all_norms.data() + begin_i, end_i - begin_i);

            // auto t_blocki_end = chrono::high_resolution_clock::now();

            for (int begin_j = 0; begin_j < total_vectors; begin_j += size_of_chunk) {
                int end_j = min(begin_j + size_of_chunk, total_vectors);

                // auto t_blockj_start = chrono::high_resolution_clock::now();
                MatrixXll block_j = load_matrix_block(matrix_file, dimension, begin_j, end_j);    
                VectorXd norms_j = Map<VectorXd>(all_norms.data() + begin_j, end_j - begin_j);
                // auto t_blockj_end = chrono::high_resolution_clock::now();

                // cout << "Processing block (" << begin_i << ":" << end_i << ") x ("
                //     << begin_j << ":" << end_j << ")" << endl;

                // auto t_dot_start = chrono::high_resolution_clock::now();
                SparseResult result = compute_sparse_dot_products_optimized(block_i, block_j, norms_i, norms_j, dimension);

                // auto t_store_start = chrono::high_resolution_clock::now();
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
        write_matrix(shard_folder, all_results, all_norms, dimension, begin_row, end_row);
        
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        #pragma omp critical
        {
            cout<<"Shard "<<shard_idx << " complete. Time: " << duration.count() << " ms" << endl;
        }
        
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
    cout<<"All shards complete. Time: " << duration.count() << " s" << endl;
    
    
    return 0;
}
