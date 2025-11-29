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
    
    // #pragma omp parallel
    // {
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
        // #pragma omp critical
        // {
    result.rows.insert(result.rows.end(), local_rows.begin(), local_rows.end());
    result.cols.insert(result.cols.end(), local_cols.begin(), local_cols.end());
    result.values.insert(result.values.end(), local_values.begin(), local_values.end());
        // }
    // }
    
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
void write_sparse_results_prev(const string& folder, 
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
        //NOTE: Assumes #genomes can be read in int32_t
        const vector<int32_t>& cols = pair.first;
        const vector<int64_t>& vals = pair.second;

        // Record the first position for this row
        index_out << row << " " << current_pos <<std::endl;
        // std::cout<<row<<" col size: "<<cols.size()<<" val size: "<<vals.size()<< endl;

        // Write column indices as differences (deltas) from previous col
        // int32_t prev_col = 0;
        int32_t prev_col = cols[0];
        bin_out.write(reinterpret_cast<const char*>(&prev_col), sizeof(int32_t));
        current_pos += sizeof(int32_t);
        for (size_t k = 1; k < cols.size(); ++k) {
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

void write_sparse_results(const string& folder, 
                         const vector<tuple<int, int, int64_t>>& results,
                         int dimension) {

    // Remove existing output folder if it exists, then create it
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }

    unordered_map<int, std::pair<vector<int>,vector<uint32_t>>> reorganized_results;
    for (const auto& [row, col, value] : results) {
        reorganized_results[row].first.push_back(col);
        reorganized_results[row].second.push_back(value);
    }

    // Write binary output: int32, vector<int32>, vector<int32>(number_of_cols, vector:diff_of_cols_with_previous_col, vector:values/2048)
    string bin_filename = folder + "matrix.bin";
    ofstream bin_out(bin_filename, ios::binary);

    // File to store the position of the first byte for each row
    // string index_filename = folder + "row_index.txt";
    string index_filename = folder + "row_index.bin";
    ofstream index_out(index_filename, ios::binary);

    // Map from row to first byte position in the binary file
    int64_t current_pos = 0;

    auto get_dot_products_vec = [&](const vector<uint32_t>& vals) {
        vector<uint32_t> dot_products_vec(vals.size());
        for (size_t k = 0; k < vals.size(); ++k) {
            dot_products_vec[k] = static_cast<uint32_t>(round(static_cast<double>(vals[k]) / dimension));
        }
        return dot_products_vec;
    };

    std::vector<int32_t> row_vec(reorganized_results.size());
    std::vector<int64_t> curr_pos_vec(reorganized_results.size());

    // Write each row's results in the new format, iterating only over rows present in reorganized_results
    int indx = 0;
    for (const auto& [row, pair] : reorganized_results) {
        //NOTE: Assumes #genomes can be read in int32_t
        const vector<int32_t>& cols = pair.first;
        const vector<uint32_t>& vals = pair.second;

        // Record the first position for this row
        // essentials::save_pod(index_out, row);
        // essentials::save_pod(index_out, current_pos);
        row_vec[indx] = row;
        curr_pos_vec[indx++] = current_pos;

        // std::cout<<indx<<": Writing row: "<< row <<" at pos: "<< current_pos << std::endl;
        
        bits::elias_fano<> ef;
        ef.encode(cols.begin(), cols.size(), cols.back()+1);
        ef.save(bin_out);
        current_pos += ef.num_bytes();

        // std::cout<<"ef bytes: "<< ef.num_bytes() << std::endl;
        // std::cout<<"col size: "<<cols.size()<<" val size: "<<vals.size()<<std::endl;

        vector<uint32_t> dot_products_vec = get_dot_products_vec(vals);
        bits::compact_vector cv;
        cv.build(dot_products_vec.begin(), dot_products_vec.size());
        cv.save(bin_out);
        current_pos += cv.num_bytes();
        // std::cout<<" cv bytes: "<< cv.num_bytes() << std::endl;
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

void write_sparse_results_rice(const string& folder, 
                         const vector<tuple<int, int, int64_t>>& results,
                         int dimension) {

    // Remove existing output folder if it exists, then create it
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }

    unordered_map<int, std::pair<vector<int>,vector<uint32_t>>> reorganized_results;
    for (const auto& [row, col, value] : results) {
        reorganized_results[row].first.push_back(col);
        reorganized_results[row].second.push_back(value);
    }

    // Write binary output: int32, vector<int32>, vector<int32>(number_of_cols, vector:diff_of_cols_with_previous_col, vector:values/2048)
    string bin_filename = folder + "matrix.bin";
    ofstream bin_out(bin_filename, ios::binary);

    // File to store the position of the first byte for each row
    // string index_filename = folder + "row_index.txt";
    string index_filename = folder + "row_index.bin";
    ofstream index_out(index_filename, ios::binary);

    // Map from row to first byte position in the binary file
    int64_t current_pos = 0;

    auto get_dot_products_vec = [&](const vector<uint32_t>& vals) {
        vector<uint32_t> dot_products_vec(vals.size());
        for (size_t k = 0; k < vals.size(); ++k) {
            dot_products_vec[k] = static_cast<uint32_t>(round(static_cast<double>(vals[k]) / dimension));
        }
        return dot_products_vec;
    };

    std::vector<uint32_t> row_vec(reorganized_results.size());
    std::vector<uint64_t> curr_pos_vec(reorganized_results.size());
    std::vector<uint32_t> start_neighbor(reorganized_results.size());

    // std::ofstream temp_out("rice_space.txt");
    // Write each row's results in the new format, iterating only over rows present in reorganized_results
    int indx = 0;
    uint64_t jac_space = 0, ngh_space = 0;
    for (const auto& [row, pair] : reorganized_results) {
        //NOTE: Assumes #genomes can be read in int32_t
        const vector<int32_t>& cols = pair.first;
        const vector<uint32_t>& vals = pair.second;

        // Record the first position for this row
        // essentials::save_pod(index_out, row);
        // essentials::save_pod(index_out, current_pos);
        row_vec[indx] = row;
        curr_pos_vec[indx] = static_cast<uint64_t>(bin_out.tellp());
        // std::streampos bin_out_pos = 
        // assert(current_pos == bin_out_pos);
        // std::cout<<"i: "<< indx <<" row: "<< row <<" at pos: "<< current_pos << std::endl;

        // std::cout<<indx<<": Writing row: "<< row <<" at pos: "<< current_pos << std::endl;
        
        start_neighbor[indx++] = cols[0];
        std::vector<uint32_t> delta_cols(cols.size()-1);
        for (size_t k = 1; k < cols.size(); ++k) {
            assert(cols[k] > cols[k-1]);
            delta_cols[k-1] = cols[k] - cols[k-1];
        }

        bits::rice_sequence<> rs;
        rs.encode(delta_cols.begin(), delta_cols.size());
        rs.save(bin_out);
        current_pos += rs.num_bytes();

        // bits::elias_fano<> ef;
        // ef.encode(cols.begin(), cols.size(), cols.back()+1);
        // ef.save(bin_out);
        // current_pos += ef.num_bytes();

        // std::cout<<"ef bytes: "<< ef.num_bytes() << std::endl;
        // std::cout<<"col size: "<<cols.size()<<" val size: "<<vals.size()<<std::endl;
        // CHECK: dot products are set as 32 bit integers
        vector<uint32_t> dot_products_vec = get_dot_products_vec(vals);
        bits::rice_sequence<> rs_dp;
        rs_dp.encode(dot_products_vec.begin(), dot_products_vec.size());
        rs_dp.save(bin_out);
        current_pos += rs_dp.num_bytes();

        // bits::compact_vector cv;
        // cv.build(dot_products_vec.begin(), dot_products_vec.size());
        // cv.save(bin_out);
        // current_pos += cv.num_bytes();
        // std::cout<<" cv bytes: "<< cv.num_bytes() << std::endl;
        // temp_out<<row<<" "<<dot_products_vec.size()<<" "<<rs_dp.num_bytes()<<" "<<rs.num_bytes()<<std::endl;
        ngh_space += rs.num_bytes();
        jac_space += rs_dp.num_bytes();
    }
    bin_out.close();
    bits::rice_sequence<> rs_rows;
    rs_rows.encode(row_vec.begin(), row_vec.size());
    rs_rows.save(index_out);

    bits::rice_sequence<> rs_pos;
    rs_pos.encode(curr_pos_vec.begin(), curr_pos_vec.size());
    rs_pos.save(index_out);
    index_out.close();

    std::string neighbor_fn = folder + "neighbor_start.bin";
    std::ofstream ngh_out(neighbor_fn, std::ios::binary);

    bits::rice_sequence<> rs_start;
    // for(int i=0; i<start_neighbor.size(); i++){
    //     std::cout<<"Neighbor start for row "<< i <<": "<< start_neighbor[i] << std::endl;
    // }
    rs_start.encode(start_neighbor.begin(), start_neighbor.size());
    rs_start.save(ngh_out);
    ngh_out.close();
    ngh_space += rs_start.num_bytes();

    std::cout<<"Jac space: "<<jac_space<<" ngh space: "<<ngh_space<<std::endl;


    // bits::compact_vector cv_rows;
    // cv_rows.build(row_vec.begin(), row_vec.size());
    // cv_rows.save(index_out);
    // bits::compact_vector cv_pos;
    // cv_pos.build(curr_pos_vec.begin(), curr_pos_vec.size());
    // cv_pos.save(index_out);
    // index_out.close();
    

    // Compress the output files using zstd and remove the originals
    string cmd1 = "zstd -f " + bin_filename + " && rm -f " + bin_filename;
    string cmd2 = "zstd -f " + index_filename + " && rm -f " + index_filename;
    string cmd3 = "zstd -f " + neighbor_fn + " && rm -f " + neighbor_fn;
    system(cmd1.c_str());
    system(cmd2.c_str());
    system(cmd3.c_str());
}


void write_sparse_results_jaccard(const string& folder, 
                         const vector<tuple<int, int, int64_t>>& results,
                         const vector<double>& all_norms_vec,
                         int dimension) {

    // Remove existing output folder if it exists, then create it
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }
    const double MULT_CONST = (1ULL << 16) - 1;
    // unordered_map<int, std::pair<vector<int>,vector<uint32_t>>> reorganized_results;
    unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint16_t> > >reorganized_results;
    
    for (const auto& [row, col, value] : results) {
        if(row == col) continue;
        double norm_curr = all_norms_vec[row];
        double norm_col = all_norms_vec[col];
        double inter_col = static_cast<double>(value)/dimension;
        double jaccard = inter_col / (norm_curr + norm_col - inter_col);
        // if(row == 32 || row == 12){
        //     std::cout<<"is: "<<inter_col<<" ns: "<<norm_curr<<" nn: "<<norm_col<<" jac: "<<jaccard<<std::endl;
        // }
        if(jaccard > 1) jaccard = 1;
        uint16_t quantized_jaccard = static_cast<uint16_t>(round(jaccard * MULT_CONST));
        reorganized_results[row].push_back(std::make_pair(col, quantized_jaccard));
    }

    // Write binary output: int32, vector<int32>, vector<int32>(number_of_cols, vector:diff_of_cols_with_previous_col, vector:values/2048)
    string bin_filename = folder + "matrix.bin";
    ofstream bin_out(bin_filename, ios::binary);

    // File to store the position of the first byte for each row
    string index_filename = folder + "row_index.bin";
    ofstream index_out(index_filename, ios::binary);

    // Map from row to first byte position in the binary file
    uint64_t current_pos = 0;
    std::vector<uint32_t> row_vec(reorganized_results.size());
    std::vector<uint64_t> curr_pos_vec(reorganized_results.size());
    std::vector<uint64_t> bytes_written_vec(reorganized_results.size());

    
    // std::ofstream temp_out("space_usage.txt");
    uint64_t jac_space = 0, ngh_space = 0;
    // Write each row's results in the new format, iterating only over rows present in reorganized_results
    int indx = 0;
    // neighbor_pair_vec<neighbor_index, jaccard_btwn_me_&_neighbor>
    for (auto& [row, neighbor_pair_vec] : reorganized_results) {
        sort(neighbor_pair_vec.begin(), neighbor_pair_vec.end(),[] (const std::pair<uint32_t, uint16_t>& a, const std::pair<uint32_t, uint16_t>& b) {
            return a.second > b.second;
        });

        std::vector<uint32_t> neighbor_indx_vec;
        std::vector<uint16_t> neighbor_jaccard_vec;
        neighbor_indx_vec.reserve(neighbor_pair_vec.size());
        neighbor_jaccard_vec.reserve(neighbor_pair_vec.size());

         for (auto& [idx, j] : neighbor_pair_vec) {
            neighbor_indx_vec.push_back(idx);
            neighbor_jaccard_vec.push_back(j);
        }

        // for(size_t i=0; i<neighbor_pair_vec.size(); i++){
        //     neighbor_indx_vec[i] = neighbor_pair_vec[i].first;
        //     neighbor_jaccard_vec[i] = neighbor_pair_vec[i].second;
        //     // if(neighbor_pair_vec[i].second >= 1) neighbor_jaccard_vec[i] = static_cast<uint16_t>(MULT_CONST);
        //     // else neighbor_jaccard_vec[i] = static_cast<uint16_t>(round(neighbor_pair_vec[i].second * MULT_CONST));
        //     // if(row == 5722){
        //     //     std::cout<<i<<" "<<neighbor_pair_vec[i].second<<" "<<neighbor_jaccard_vec[i]<<std::endl;
        //     // }
        // }

        // Record this row index and its stored position 
        uint64_t current_pos = static_cast<uint64_t>(bin_out.tellp());
        row_vec[indx] = row;
        curr_pos_vec[indx] = current_pos;

        // std::cout<<"i: "<< indx <<" row: "<< row <<" at pos: "<< current_pos << std::endl;

        // std::cout<<indx<<": Writing row: "<< row <<" at pos: "<< current_pos << std::endl;

        std::vector<uint16_t> ngh_jcrd_delta_vec(neighbor_jaccard_vec.size()-1);
        uint16_t top_jaccard = neighbor_jaccard_vec[0];
        for(size_t i=1; i < neighbor_jaccard_vec.size(); i++){
            // as sorted in descending order
            if(neighbor_jaccard_vec[i-1] < neighbor_jaccard_vec[i]){
                std::cout<<i<<" "<<row<<" "<<neighbor_jaccard_vec[i-1] <<" "<< neighbor_jaccard_vec[i]<<std::endl;
            }
            assert(neighbor_jaccard_vec[i-1] >= neighbor_jaccard_vec[i]);
            ngh_jcrd_delta_vec[i-1] = neighbor_jaccard_vec[i-1] - neighbor_jaccard_vec[i];
        }
        // std::cout<<"indx: "<<indx<<" row: "<<row<<" ca: "<<current_pos
        //     <<" tj: "<<top_jaccard<<" so: "<<sizeof(top_jaccard)
        //     <<std::endl;
        bin_out.write(reinterpret_cast<const char*>(&top_jaccard), sizeof(top_jaccard));
        // bin_out.flush();
        // current_pos += sizeof(top_jaccard);
        // std::cout<<"ca: "<<current_pos<<std::endl;
        bits::rice_sequence<> delta_jaccard_rs;
        delta_jaccard_rs.encode(ngh_jcrd_delta_vec.begin(), ngh_jcrd_delta_vec.size());
        delta_jaccard_rs.save(bin_out);
        // current_pos += delta_jaccard_rs.num_bytes();
        // std::cout<<"ca: "<<current_pos<<std::endl;

        // bits::rice_sequence<> ngh_rs;
        // ngh_rs.encode(neighbor_indx_vec.begin(), neighbor_indx_vec.size());
        // ngh_rs.save(bin_out);

        bits::compact_vector cv_ngh;
        cv_ngh.build(neighbor_indx_vec.begin(), neighbor_indx_vec.size());
        cv_ngh.save(bin_out);

        // std::vector<uint8_t> encoded_ngh_indx_vec(streamvbyte_max_compressedbytes(neighbor_indx_vec.size()));
        // uint64_t bytes_written = streamvbyte_encode(neighbor_indx_vec.data(), neighbor_indx_vec.size(), 
        //             encoded_ngh_indx_vec.data());
        // bytes_written_vec[indx++] = bytes_written;
        // bin_out.write(reinterpret_cast<const char*>(encoded_ngh_indx_vec.data()), bytes_written);  
        // temp_out<<row<<" "<<neighbor_indx_vec.size()<<" "<<delta_jaccard_rs.num_bytes()<<" "<<bytes_written<<" "<<std::endl;
        
        
        jac_space += delta_jaccard_rs.num_bytes() + sizeof(top_jaccard);
        // ngh_space += bytes_written;
        // ngh_space += ngh_rs.num_bytes();
        ngh_space += cv_ngh.num_bytes();
        
    }
    bin_out.flush();     
    bin_out.close();
    

    // temp_out.close();

    bits::compact_vector cv_rows;
    cv_rows.build(row_vec.begin(), row_vec.size());
    cv_rows.save(index_out);

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

    // std::string vbyte_fn = folder + "vbyte.bin";
    // std::ofstream vbyte_out(vbyte_fn, std::ios::binary);
    // bits::compact_vector cv_vb; // Compact Vector Variable Byte written
    // cv_vb.build(bytes_written_vec.begin(), bytes_written_vec.size());
    // cv_vb.save(vbyte_out);
    // vbyte_out.close();
    // ngh_space += cv_vb.num_bytes();
    std::cout<<"Jac space: "<<jac_space<<" ngh space: "<<ngh_space<<std::endl;

    // Compress the output files using zstd and remove the originals
    string cmd1 = "zstd -f " + bin_filename + " && rm -f " + bin_filename;
    string cmd2 = "zstd -f " + index_filename + " && rm -f " + index_filename;
    // string cmd3 = "zstd -f " + vbyte_fn + " && rm -f " + vbyte_fn;
    system(cmd1.c_str());
    system(cmd2.c_str());
    // system(cmd3.c_str());
}

void write_sparse_results_jaccard_wo_sort(const string& folder, 
                         const vector<tuple<int, int, int64_t>>& results,
                         const vector<double>& all_norms_vec,
                         int dimension) {

    // Remove existing output folder if it exists, then create it
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }
    const double MULT_CONST = (1ULL << 8) - 1;
    // unordered_map<int, std::pair<vector<int>,vector<uint32_t>>> reorganized_results;
    unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint16_t> > >reorganized_results;
    // std::ofstream jac_os("jaccards.txt");
    for (const auto& [row, col, value] : results) {
        // if(row == col) continue;
        double norm_curr = all_norms_vec[row];
        double norm_col = all_norms_vec[col];
        double inter_col = static_cast<double>(value)/dimension;
        double jaccard = inter_col / (norm_curr + norm_col - inter_col);
        if(jaccard > 1) jaccard = 1;
        uint16_t quantized_jaccard = static_cast<uint16_t>(round(jaccard * MULT_CONST));
        // if(row == 9599){
        //     jac_os<<"is: "<<inter_col<<" ns: "<<norm_curr<<" nn: "<<norm_col
        //         <<" jac: "<<jaccard<<" qj: "<<quantized_jaccard
        //         <<std::endl;
        // }
        reorganized_results[row].push_back(std::make_pair(col, quantized_jaccard));
    }

    // Write binary output: int32, vector<int32>, vector<int32>(number_of_cols, vector:diff_of_cols_with_previous_col, vector:values/2048)
    string bin_filename = folder + "matrix.bin";
    ofstream bin_out(bin_filename, ios::binary);

    // File to store the position of the first byte for each row
    string index_filename = folder + "row_index.bin";
    ofstream index_out(index_filename, ios::binary);

    // Map from row to first byte position in the binary file
    // uint64_t current_pos = 0;
    std::vector<uint32_t> row_vec(reorganized_results.size());
    std::vector<uint64_t> curr_pos_vec(reorganized_results.size());
    std::vector<uint32_t> start_neighbor(reorganized_results.size());
    // std::vector<uint64_t> bytes_written_vec(reorganized_results.size());

    
    // std::ofstream temp_out("space_usage.txt");
    uint64_t jac_space = 0, ngh_space = 0;
    // Write each row's results in the new format, iterating only over rows present in reorganized_results
    int indx = 0;
    // neighbor_pair_vec<neighbor_index, jaccard_btwn_me_&_neighbor>
    for (auto& [row, neighbor_pair_vec] : reorganized_results) {
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
        row_vec[indx] = row;
        curr_pos_vec[indx] = current_pos;
    
        start_neighbor[indx++] = neighbor_indx_vec[0];
        // start_neighbor[indx] = neighbor_indx_vec[0]; //FIXME: indx++ if not vbyte
        
        std::vector<uint64_t> delta_cols(neighbor_indx_vec.size()-1);
        for (size_t k = 1; k < neighbor_indx_vec.size(); ++k) {
            assert(neighbor_indx_vec[k] > neighbor_indx_vec[k-1]);
            delta_cols[k-1] = neighbor_indx_vec[k] - neighbor_indx_vec[k-1];
        }

        bits::compact_vector cv_jc;
        cv_jc.build(neighbor_jaccard_vec.begin(), neighbor_jaccard_vec.size());
        cv_jc.save(bin_out);
        
        jac_space += cv_jc.num_bytes();
        
        assert(neighbor_jaccard_vec.size() >= 1);
        
        if(neighbor_jaccard_vec.size() == 1) continue;
        
        bits::rice_sequence<> rs_delta;
        rs_delta.encode(delta_cols.begin(), delta_cols.size());
        rs_delta.save(bin_out);

        

        
        
        

        // if(row == 9599){
        //     std::cout<<row<<" "<<current_pos
        //         <<" "<<rs_delta.size()+1<<" "<<neighbor_indx_vec.size()<<" "<< neighbor_indx_vec[0]
        //     <<std::endl;
        // }

        // bits::rice_sequence<> rs_jac;
        // rs_jac.encode(neighbor_jaccard_vec.begin(), neighbor_jaccard_vec.size());
        // rs_jac.save(bin_out);

        
        // jac_space += rs_jac.num_bytes();
        // ngh_space += bytes_written;
        // ngh_space += ngh_rs.num_bytes();
        ngh_space += rs_delta.num_bytes();
        // ngh_space += cv_delta.num_bytes();
        // ngh_space += bytes_written;
        
    }
    bin_out.flush();     
    bin_out.close();
    

    // temp_out.close();

    bits::compact_vector cv_rows;
    cv_rows.build(row_vec.begin(), row_vec.size());
    cv_rows.save(index_out);

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
    ngh_space += rs_start.num_bytes();

    // std::string vbyte_fn = folder + "vbyte.bin";
    // std::ofstream vbyte_out(vbyte_fn, std::ios::binary);
    // bits::compact_vector cv_vb; // Compact Vector Variable Byte written
    // cv_vb.build(bytes_written_vec.begin(), bytes_written_vec.size());
    // cv_vb.save(vbyte_out);
    // vbyte_out.close();
    // ngh_space += cv_vb.num_bytes();

    // std::string vbyte_fn = folder + "vbyte.bin";
    // std::ofstream vbyte_out(vbyte_fn, std::ios::binary);
    // bits::compact_vector cv_vb; // Compact Vector Variable Byte written
    // cv_vb.build(bytes_written_vec.begin(), bytes_written_vec.size());
    // cv_vb.save(vbyte_out);
    // vbyte_out.close();
    // ngh_space += cv_vb.num_bytes();
    std::cout<<"Jac space: "<<jac_space<<" ngh space: "<<ngh_space<<std::endl;

    // Compress the output files using zstd and remove the originals
    // string cmd1 = "zstd -f " + bin_filename + " && rm -f " + bin_filename;
    // string cmd2 = "zstd -f " + index_filename + " && rm -f " + index_filename;
    // string cmd3 = "zstd -f " + neighbor_fn + " && rm -f " + neighbor_fn;
    // system(cmd1.c_str());
    // system(cmd2.c_str());
    // system(cmd3.c_str());
}


int main(int argc, char* argv[]) {
    // Argument parsing using clipp
    string db_folder, matrix_file;
    int dimension = 0;
    double max_memory_gb = 0.0;
    int num_threads = 1;
    string output_folder;
    int num_shards = 1;
    int shard_idx = 0;
    int strategy = 0; // 0=random projections, 1=minHashes
    int start_shard = 0;
    int end_shard = num_shards;

    bool show_help = false;

    auto cli = (
        clipp::option("--db") & clipp::value("file", db_folder),
        clipp::option("--dimension") & clipp::value("int", dimension),
        clipp::option("--max_memory_gb") & clipp::value("float", max_memory_gb),
        clipp::option("--num_threads") & clipp::value("int", num_threads),
        clipp::option("--output_folder") & clipp::value("folder", output_folder),
        clipp::option("--num_shards") & clipp::value("int", num_shards),
        clipp::option("--shard_idx") & clipp::value("int", shard_idx),
        clipp::option("--strategy") & clipp::value("int", strategy),
        clipp::option("--start_shard") & clipp::value("int", start_shard),
        clipp::option("--end_shard") & clipp::value("int", end_shard),
        clipp::option("--help").set(show_help)
    );

    if (!clipp::parse(argc, argv, cli) || show_help) {
        cout << "Usage:\n"
             << clipp::usage_lines(cli, argv[0]) << endl;
        cout << "\n--strategy 0=random projections (default), 1=minHashes\n";
        return show_help ? 0 : 1;
    }

    if (db_folder.empty() || dimension <= 0 || max_memory_gb <= 0.0 || num_threads <= 0 || output_folder.empty() || num_shards <= 0 || shard_idx < 0 || shard_idx >= num_shards) {
        cerr << "Missing or invalid arguments. Use --help for usage." << endl;
        return 1;
    }

    // Ensure output folder ends with '/'
    if (!output_folder.empty() && output_folder.back() != '/' && output_folder.back() != '\\') {
        output_folder += '/';
    }

    if (!db_folder.empty() && db_folder.back() != '/' && db_folder.back() != '\\') {
        output_folder += '/';
    }

    matrix_file = db_folder + "vectors.bin";

    string norms_file = db_folder + "vector_norms.txt";
    if (!fs::exists(norms_file)) {
        cerr << "Error: Required file 'vector_norms.txt' not found in output folder: " << db_folder << endl;
        return 1;
    }

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

    auto start_time = chrono::high_resolution_clock::now();
    
    // int outer_threads = min(num_threads, min(8, num_shards));
    // int inner_threads = max(1, num_threads / outer_threads);
    // omp_set_nested(1);
    // omp_set_max_active_levels(2);

    // omp_set_num_threads(outer_threads);
    // #pragma omp parallel for schedule(dynamic)
    omp_set_num_threads(num_threads);
    // for(int i=start_shard; i<end_shard; i++){
        // omp_set_num_threads(inner_threads);

        // shard_idx = i;
    string shard_folder = output_folder + "shard_" + to_string(shard_idx) + "/";
    if (!fs::exists(shard_folder)) {
        fs::create_directories(shard_folder);
    }

    // Compute row range for this shard
    int rows_per_shard = (total_vectors + num_shards - 1) / num_shards;
    int begin_row = shard_idx * rows_per_shard;
    int end_row = min(begin_row + rows_per_shard, total_vectors);

    // begin_row = 9599;
    // end_row = begin_row + 2;

    cout << "Shard " << shard_idx << " processing rows " << begin_row << " to " << end_row << endl;

    vector<tuple<int, int, int64_t>> all_results;

    for (int begin_i = begin_row; begin_i < end_row; begin_i += size_of_chunk) {
        int end_i = min(begin_i + size_of_chunk, end_row);

        // auto t_blocki_start = chrono::high_resolution_clock::now();
        MatrixXi block_i = load_matrix_block(matrix_file, dimension, begin_i, end_i);
        VectorXd norms_i = Map<VectorXd>(all_norms.data() + begin_i, end_i - begin_i);

        // auto t_blocki_end = chrono::high_resolution_clock::now();

        for (int begin_j = 0; begin_j < total_vectors; begin_j += size_of_chunk) {
            int end_j = min(begin_j + size_of_chunk, total_vectors);

            // auto t_blockj_start = chrono::high_resolution_clock::now();
            MatrixXi block_j = load_matrix_block(matrix_file, dimension, begin_j, end_j);
            VectorXd norms_j = Map<VectorXd>(all_norms.data() + begin_j, end_j - begin_j);
            // auto t_blockj_end = chrono::high_resolution_clock::now();

            // cout << "Processing block (" << begin_i << ":" << end_i << ") x ("
            //     << begin_j << ":" << end_j << ")" << endl;

            // auto t_dot_start = chrono::high_resolution_clock::now();
            SparseResult result;
            if (strategy == 0){ //random projections
                result = compute_sparse_dot_products_optimized(block_i, block_j, norms_i, norms_j, dimension);
            }
            else{ //minHashes
                result = compute_jaccard_with_MinHash(block_i, block_j, dimension);
            }

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

    // Write results to the shard subfolder
    // write_sparse_results_prev(shard_folder, all_results, dimension);
    // write_sparse_results(shard_folder, all_results, dimension);
    // write_sparse_results_rice(shard_folder, all_results, dimension);
    // write_sparse_results_jaccard(shard_folder, all_results, all_norms ,dimension);
    write_sparse_results_jaccard_wo_sort(shard_folder, all_results, all_norms ,dimension);
    // }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Total computation time: " << duration.count() << " ms" << endl;
    
    return 0;
}
