#include "read_pc_mat.h"
#include "clipp.h"
#include "cnpy.h"
#include <highfive/H5File.hpp>
#include <memory>
#include <omp.h> 
#include <cassert>
#include <cmath>
#include <atomic>

namespace fs = std::filesystem;


void show_error_and_exit(std::string msg){
    std::cerr<<msg<<std::endl;
    std::cerr<<"Aborting...\n";
    exit(1);
}

double roundUpToTwoDecimals(double num) {
  return std::ceil(num * 100.0) / 100.0;
}

std::pair<double, std::string> get_time_unit(double total_time){
    if(total_time < 60){
        // return std::to_string(static_cast<uint64_t>( std::ceil(total_time)))+"\t seconds";
        return std::make_pair(total_time, "seconds");
        // return std::to_string(roundUpToTwoDecimals(total_time))+"\t seconds";
    }
    else if(total_time < 60*60){
        total_time/=60.0;
        // return std::to_string(static_cast<uint64_t>( std::ceil(total_time)))+"\t minutes";
        return std::make_pair(total_time, "minutes");
    }
    else{
        total_time/=(60.0*60);
        // return std::to_string(static_cast<uint64_t>( std::ceil(total_time)))+"\t hours";
        return std::make_pair(total_time, "hours");
    }
}

std::pair<std::string, std::string> split_path(const std::string& fullpath) {
    size_t pos = fullpath.find_last_of("/\\");
    if (pos == std::string::npos) {
        return {fullpath, "./"};  // no directory
    }

    std::string filename = fullpath.substr(pos + 1);
    std::string parent   = fullpath.substr(0, pos);
    return {filename, parent};
}


void query_nearest_neighbors(
    std::string matrix_folder, std::string db_folder, std::string query_file,
    std::vector<std::string>& query_ids_str, bool write_to_file, 
    bool show_all_neighbors, int64_t top_n, uint32_t batch_size, std::string out_fn, 
    std::string sep, bool print_to_screen, int num_threads
) {
    std::vector<std::string> identifiers;
    std::unordered_map<std::string, int> id_to_index = pc_mat::load_vector_identifiers(db_folder, identifiers);
    
    std::vector<std::string> query_id_vec;    
    std::vector<uint32_t> queries;

    if (!query_file.empty()) {
        queries = pc_mat::read_queries_from_file(query_file, id_to_index, query_id_vec);
    } else if (!query_ids_str.empty()) {
        for (const std::string& query_str : query_ids_str) {
            int index = pc_mat::parse_query_to_index(query_str, id_to_index);
            if (index >= 0) {
                queries.push_back(index);
            }
        }
    } else {
        show_error_and_exit("Error: No queries specified. Use --query_file, --query_ids");
    }

    if (queries.empty()) {
        show_error_and_exit("Error: No valid queries found");
    }

    std::vector<float> vector_norms;
    pc_mat::load_vector_norms(db_folder, vector_norms);

    int total_vectors = identifiers.size();
    std::cout << "Total vectors loaded: " << total_vectors << std::endl << std::endl;
    if (total_vectors <= 0) {
        show_error_and_exit("Error: Could not determine total number of vectors");
    }
    
    auto file_info = split_path(out_fn);
    std::string fname = file_info.first;
    std::string out_file_path = file_info.second;
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    size_t total_queries = queries.size();
    size_t num_batches = (total_queries + batch_size - 1) / batch_size;
    std::atomic<size_t> completed_queries(0);

    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t b = 0; b < num_batches; ++b) {
        size_t start_indx = b * batch_size;
        size_t end_indx = std::min(start_indx + batch_size, total_queries);

        std::vector<int32_t> sub_queries(
            queries.begin() + start_indx, 
            queries.begin() + end_indx
        );

        std::vector<pc_mat::Result> batch_results = pc_mat::query(
            matrix_folder, sub_queries, vector_norms, identifiers
        );

        for (size_t i = 0; i < batch_results.size(); ++i) {
            const pc_mat::Result& res = batch_results[i];
            
            if (print_to_screen) {
                #pragma omp critical
                {
                    std::cout << "Query: " << res.self_id << " #Neighbors: " << res.neighbor_ids.size() << std::endl;
                }
            }
            
            // File Output: Independent files, no lock needed for writing
            std::ofstream out;
            if (write_to_file) {
                if(!res.self_id.empty()){
                    std::string nfn = out_file_path + "/" + res.self_id + "_" + fname;
                    #pragma omp critical
                    {
                        std::cout << "Writing in file: " << nfn << std::endl << std::endl;
                    }
                    out.open(nfn.c_str());
                    out << "ID" << sep << "Jaccard\n";
                }
                else{
                    std::cout<<"Warning: Empty identifier for query index "<< start_indx + i <<". Skipping file write.\n";
                }
                
            }

            int64_t num_neighbors_to_show = show_all_neighbors ? 
                        res.neighbor_ids.size()
                        : std::min<int64_t>(top_n, res.neighbor_ids.size());

            if (print_to_screen) {
                #pragma omp critical
                {
                    if(!res.self_id.empty()){
                        std::cout << "Top " << num_neighbors_to_show << " neighbors for " << res.self_id << ":\n";
                        for (size_t j = 0; j < num_neighbors_to_show; ++j) {
                            std::cout << j + 1 << ". Neighbor: " << res.neighbor_ids[j]
                                    << " Jaccard Similarity: " << res.jaccard_similarities[j] << std::endl;
                        }
                        std::cout << std::endl;
                    }
                    else{
                        std::cout<<"Warning: Empty identifier for query index "<< start_indx + i <<". Skipping screen output.\n";
                    }
                }
            }

            if (write_to_file && !res.self_id.empty()) {
                for (size_t j = 0; j < num_neighbors_to_show; ++j) {
                    out << res.neighbor_ids[j] << sep << res.jaccard_similarities[j] << std::endl;
                }
                out.close();
            }
        }
        
        size_t current_completed = completed_queries.fetch_add(sub_queries.size()) + sub_queries.size();
        
        if (b % num_threads == 0 || current_completed == total_queries) {
            #pragma omp critical 
            {
                std::cout << "--------- Progress: " << current_completed << " / " << total_queries << " queries processed ---------" << std::endl;
            }
        }
    }    

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_total - start_total;
    auto time_unit = get_time_unit(elapsed.count());

    std::cout << "\nAll Queries completed in " << std::fixed << std::setprecision(2) 
              << time_unit.first << "\t" << time_unit.second << "\n" << std::endl;
}

void query_sliced_matrix(
    std::string matrix_folder, std::string db_folder, std::string row_file, std::string col_file,
    bool write_to_file, std::string out_fn, uint32_t batch_size, bool print_to_screen, std::string sep,
    std::string file_extension, int num_threads
) {
    // --- 1. Load Data ---
    std::vector<std::string> identifiers;
    std::unordered_map<std::string, int> id_to_index = pc_mat::load_vector_identifiers(db_folder, identifiers);

    std::vector<uint32_t> row_query_vec, col_query_vec;
    std::vector<std::string> row_vec, col_vec;

    row_query_vec = pc_mat::read_queries_from_file(row_file, id_to_index, row_vec);
    col_query_vec = pc_mat::read_queries_from_file(col_file, id_to_index, col_vec);

    if (row_query_vec.empty() || col_query_vec.empty()) {
        show_error_and_exit("Empty row or col accessions.");
    }

    std::vector<float> vector_norms;
    pc_mat::load_vector_norms(db_folder, vector_norms);

    int total_vectors = identifiers.size();
    std::cout << "Total vectors loaded: " << total_vectors << std::endl << std::endl;
    if (total_vectors <= 0) {
        show_error_and_exit("Error: Could not determine total number of vectors");
    }

    // --- 2. Setup Output Files ---
    std::ofstream out; 
    std::unique_ptr<HighFive::File> hf_ptr;
    HighFive::DataSetCreateProps props;

    // Initialize files (Open once, keep open or append later)
    if (write_to_file) {
        std::cout << "Writing in file: " << out_fn << std::endl << std::endl;
        if (file_extension == "csv" || file_extension == "tsv") {
            out.open(out_fn.c_str());
            // Write Header
            out << "Accession" << sep;
            for (size_t i = 0; i < col_vec.size(); i++) {
                out << col_vec[i] << sep;
            }
            out << "\n";
        } else if (file_extension == "h5") {
            hf_ptr = std::make_unique<HighFive::File>(out_fn, HighFive::File::Overwrite);
            hsize_t safe_chunk_size = std::min(static_cast<hsize_t>(col_vec.size()),
                                               static_cast<hsize_t>(16384));
            props.add(HighFive::Chunking(std::vector<hsize_t>{safe_chunk_size}));
            props.add(HighFive::Shuffle());
            props.add(HighFive::Deflate(9)); // how much compression
        }
        // } else if (file_extension == "npy" || file_extension == "npz") {
        //     std::vector<float> res;
        //     cnpy::npy_save(out_fn, res.data(), {1, res.size()}, "w");
        //     // Initialize/Truncate file so we can safely append later
        //     // std::ofstream clear_file(out_fn, std::ios::out | std::ios::trunc);
        //     // clear_file.close();
        // }
    }

    if (print_to_screen) {
        std::cout << "Accession\t";
        for (size_t i = 0; i < col_vec.size(); i++) {
            std::cout << col_vec[i] << "\t";
        }
        std::cout << "\n";
    }

    // --- 3. Parallel Processing (Block/Wavefront Strategy) ---
    auto start_total = std::chrono::high_resolution_clock::now();
    size_t total_rows = row_query_vec.size();
    
    // Determine the step size for the outer loop: n_threads * batch_size
    // This is the "Super Batch" size
    size_t block_step = (size_t)num_threads * batch_size;
    size_t current_block_start = 0;

    // Buffer to hold results from all threads for the current super-batch
    // Index: [thread_id][row_index_within_batch][col_index]
    std::vector<std::vector<std::vector<float>>> thread_buffers(num_threads);

    omp_set_num_threads(num_threads);

    while (current_block_start < total_rows) {
        
        // --- A. Parallel Compute Phase ---
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t my_batch_start = current_block_start + tid * batch_size;

            if (my_batch_start < total_rows) {
                size_t my_batch_end = std::min(my_batch_start + batch_size, total_rows);

                std::vector<uint32_t> row_sub_queries(
                    row_query_vec.begin() + my_batch_start, 
                    row_query_vec.begin() + my_batch_end
                );

                // Compute and store in thread-specific buffer
                // This runs in parallel, no locks needed as they write to different indices of thread_buffers
                thread_buffers[tid] = pc_mat::query_sliced(
                    matrix_folder, row_sub_queries, col_query_vec, total_vectors, vector_norms
                );
            } else {
                // Thread has no work (e.g., end of file), clear buffer
                thread_buffers[tid].clear();
            }
        } // Implicit Barrier: wait for all threads to finish computing

        // --- B. Sequential Write Phase ---
        // Iterate through threads in order to maintain row sequence
        for (int t = 0; t < num_threads; ++t) {
            if (thread_buffers[t].empty()) continue;

            size_t my_batch_start = current_block_start + t * batch_size;
            std::vector<std::vector<float>>& batch_results = thread_buffers[t];

            for (size_t i = 0; i < batch_results.size(); ++i) {
                std::vector<float>& res = batch_results[i];
                size_t actual_row_idx = my_batch_start + i;

                // 1. Screen Output
                if (print_to_screen) {
                    std::cout << row_vec[actual_row_idx] << "\t";
                    for (float val : res) std::cout << val << "\t";
                    std::cout << std::endl;
                }

                // 2. File Output
                if (write_to_file) {
                    if (file_extension == "csv" || file_extension == "tsv") {
                        out << row_vec[actual_row_idx] << sep;
                        for (size_t j = 0; j < res.size(); ++j) {
                            out << res[j] << sep;
                        }
                        out << "\n";
                    } 
                    else if (file_extension == "h5") {
                        hf_ptr->createDataSet(row_vec[actual_row_idx], res, props);
                    } 
                    else if (file_extension == "npy" || file_extension == "npz") {
                        if (current_block_start == 0 && t == 0 && i == 0)
                            cnpy::npy_save(out_fn, res.data(), {1, res.size()}, "w");
                        else
                            cnpy::npy_save(out_fn, res.data(), {1, res.size()}, "a");
                    }
                }
            }
            // Clear memory for this thread immediately after writing
            batch_results.clear(); 
        }
        
        std::cout << "--------- Completed\t" 
                  << std::min(current_block_start + block_step, total_rows) 
                  << "\trows in\t";
        auto mid_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = mid_total - start_total;
        auto time_unit = get_time_unit(elapsed.count());
        std::cout << std::fixed << std::setprecision(2) << time_unit.first << "\t" 
                  << time_unit.second << " ---------\n";

        current_block_start += block_step;
    }

    // --- 4. Cleanup ---
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_total - start_total;
    auto time_unit = get_time_unit(elapsed.count());

    std::cout << "\nAll Queries completed in " << std::fixed << std::setprecision(2) 
              << time_unit.first << "\t" << time_unit.second << "\n" << std::endl;

    if (write_to_file && (file_extension == "csv" || file_extension == "tsv")) {
        out.close();
    }
}

void filter_matrix(std::string matrix_folder, std::string db_folder, std::string store_folder, double filter, int num_threads){
    std::vector<float> vector_norms;
    pc_mat::load_vector_norms(db_folder, vector_norms);

    uint64_t total_vectors = pc_mat::get_total_vectors(db_folder);
    uint64_t num_shards = pc_mat::discover_shards(matrix_folder);
    uint64_t rows_per_shard = (total_vectors + num_shards - 1) / num_shards;

    auto start_total = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    
    for(size_t shard_idx=0; shard_idx < num_shards; shard_idx++){
        auto shard_start = std::chrono::high_resolution_clock::now();
        std::string shard_folder = matrix_folder + "/shard_" + std::to_string(shard_idx);
        std::string new_shard_folder = store_folder + "/shard_" + std::to_string(shard_idx);
        fs::path dir_path = new_shard_folder;
        fs::create_directories(dir_path);
        std::cout<<"Writing filtered matrix in "<<new_shard_folder<<std::endl;

        uint64_t start_row = shard_idx * rows_per_shard;
        uint64_t end_row = min(start_row + rows_per_shard, total_vectors);
        if(start_row < end_row){
            pc_mat::filter_matrix_for_shard(shard_folder, new_shard_folder, start_row, end_row, filter);
        }
            
        auto shard_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = shard_end - shard_start;
        auto shard_time = get_time_unit(elapsed.count());
        #pragma omp critical
        {
            std::cout << "Shard " << shard_idx
                      << " completed in "
                      << std::fixed << std::setprecision(2)
                      << shard_time.first << " "
                      << shard_time.second << '\n';
        }
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_total - start_total;
    auto time_unit = get_time_unit(elapsed.count());

    std::cout << "\nAll shards completed in " << std::fixed << std::setprecision(2) 
              << time_unit.first << "\t" << time_unit.second << "\n" << std::endl;
}



void store_only_top_n_matrix(std::string matrix_folder, std::string db_folder, std::string store_folder, uint32_t num_acc, int num_threads){
    uint64_t total_vectors = pc_mat::get_total_vectors(db_folder);
    uint64_t num_shards = pc_mat::discover_shards(matrix_folder);
    uint64_t rows_per_shard = (total_vectors + num_shards - 1) / num_shards;

    auto start_total = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    
    for(size_t shard_idx=0; shard_idx < num_shards; shard_idx++){
        auto shard_start = std::chrono::high_resolution_clock::now();
        std::string shard_folder = matrix_folder + "/shard_" + std::to_string(shard_idx);
        std::string new_shard_folder = store_folder + "/shard_" + std::to_string(shard_idx);
        fs::path dir_path = new_shard_folder;
        fs::create_directories(dir_path);
        #pragma omp critical
        {
            std::cout<<"Writing filtered matrix in "<<new_shard_folder<<std::endl;
        }
        
        uint64_t start_row = shard_idx * rows_per_shard;
        uint64_t end_row = min(start_row + rows_per_shard, total_vectors);
        if(start_row < end_row){
            pc_mat::store_only_top_n_matrix_for_shard(shard_folder, new_shard_folder, start_row, end_row, num_acc);
        }
            
        auto shard_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = shard_end - shard_start;
        auto shard_time = get_time_unit(elapsed.count());
        #pragma omp critical
        {
            std::cout << "Shard " << shard_idx
                      << " completed in "
                      << std::fixed << std::setprecision(2)
                      << shard_time.first << " "
                      << shard_time.second << '\n';
        }
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_total - start_total;
    auto time_unit = get_time_unit(elapsed.count());

    std::cout << "\nAll shards completed in " << std::fixed << std::setprecision(2) 
              << time_unit.first << "\t" << time_unit.second << "\n" << std::endl;
}


void store_only_top_n_wfil_matrix(std::string matrix_folder, std::string db_folder, std::string store_folder, uint32_t num_acc, double filter, int num_threads){
    uint64_t total_vectors = pc_mat::get_total_vectors(db_folder);
    uint64_t num_shards = pc_mat::discover_shards(matrix_folder);
    uint64_t rows_per_shard = (total_vectors + num_shards - 1) / num_shards;

    auto start_total = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    
    for(size_t shard_idx=0; shard_idx < num_shards; shard_idx++){
        auto shard_start = std::chrono::high_resolution_clock::now();
        std::string shard_folder = matrix_folder + "/shard_" + std::to_string(shard_idx);
        std::string new_shard_folder = store_folder + "/shard_" + std::to_string(shard_idx);
        fs::path dir_path = new_shard_folder;
        fs::create_directories(dir_path);
        #pragma omp critical
        {
            std::cout<<"Writing filtered matrix in "<<new_shard_folder<<std::endl;
        }
        
        uint64_t start_row = shard_idx * rows_per_shard;
        uint64_t end_row = min(start_row + rows_per_shard, total_vectors);
        if(start_row < end_row){
            pc_mat::store_only_top_n_wfil_matrix_for_shard(shard_folder, new_shard_folder, start_row, end_row, num_acc, filter);
        }
            
        auto shard_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = shard_end - shard_start;
        auto shard_time = get_time_unit(elapsed.count());
        #pragma omp critical
        {
            std::cout << "Shard " << shard_idx
                      << " completed in "
                      << std::fixed << std::setprecision(2)
                      << shard_time.first << " "
                      << shard_time.second << '\n';
        }
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_total - start_total;
    auto time_unit = get_time_unit(elapsed.count());

    std::cout << "\nAll shards completed in " << std::fixed << std::setprecision(2) 
              << time_unit.first << "\t" << time_unit.second << "\n" << std::endl;
}



void write_neighbor_from_matrix(std::string matrix_folder, std::string db_folder,int num_threads){
    
    uint64_t total_vectors = pc_mat::get_total_vectors(db_folder);
    uint64_t num_shards = pc_mat::discover_shards(matrix_folder);
    uint64_t rows_per_shard = (total_vectors + num_shards - 1) / num_shards;

    auto start_total = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    
    for(size_t shard_idx=0; shard_idx < num_shards; shard_idx++){
        auto shard_start = std::chrono::high_resolution_clock::now();
        std::string shard_folder = matrix_folder + "/shard_" + std::to_string(shard_idx);

        uint64_t start_row = shard_idx * rows_per_shard;
        uint64_t end_row = min(start_row + rows_per_shard, total_vectors);
        if(start_row < end_row){
            pc_mat::save_neighbors_for_shard(shard_folder, start_row, end_row);
        }
            
        auto shard_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = shard_end - shard_start;
        auto shard_time = get_time_unit(elapsed.count());
        #pragma omp critical
        {
            std::cout << "Shard " << shard_idx
                      << " completed in "
                      << std::fixed << std::setprecision(2)
                      << shard_time.first << " "
                      << shard_time.second << '\n';
        }
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_total - start_total;
    auto time_unit = get_time_unit(elapsed.count());

    std::cout << "\nAll shards completed in " << std::fixed << std::setprecision(2) 
              << time_unit.first << "\t" << time_unit.second << "\n" << std::endl;
}


std::string get_whitespace_removed(std::string &s){
    const std::string WHITESPACE = " \n\r\t\f\v";

    size_t end = s.find_last_not_of(WHITESPACE);
    if (end != std::string::npos) {
        s.erase(end + 1);
    } else {
        s.clear(); // The string is entirely whitespace
    }
    return s;
}

void update_matrix_from_list(std::string matrix_folder, std::string db_folder, std::string store_folder, std::string acc_db_folder, int num_threads){
    std::vector<float> vector_norms;
    pc_mat::load_vector_norms(db_folder, vector_norms);

    std::vector<std::string> identifiers_prev;
    std::unordered_map<std::string, int> id_to_index_prev = pc_mat::load_vector_identifiers(db_folder, identifiers_prev);

    
    std::vector<std::string> identifiers_new;
    std::unordered_map<std::string, int> id_to_index_new = pc_mat::load_vector_identifiers(acc_db_folder, identifiers_new);

    uint64_t total_vectors_prev = pc_mat::get_total_vectors(db_folder);
    uint64_t num_shards = pc_mat::discover_shards(matrix_folder);
    uint64_t rows_per_shard_prev = (total_vectors_prev + num_shards - 1) / num_shards;

    std::vector<uint32_t> acc_vec;
    
    std::vector<uint32_t> new_index_to_prev_index_vec(identifiers_new.size()); //works like a map
    
    for(size_t i=0; i<identifiers_new.size(); i++){
        uint32_t prev_index = id_to_index_prev.at(identifiers_new[i]);
        acc_vec.push_back(prev_index);
        new_index_to_prev_index_vec[i] = prev_index;

    }
   

    uint64_t total_vectors_new = acc_vec.size();
    uint64_t rows_per_shard_new = (total_vectors_new + num_shards - 1) / num_shards;
    
    auto start_total = std::chrono::high_resolution_clock::now();

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    
    for(size_t shard_idx=0; shard_idx < num_shards; shard_idx++){
        auto shard_start = std::chrono::high_resolution_clock::now();
        std::string shard_folder = matrix_folder + "/shard_" + std::to_string(shard_idx);
        std::string new_shard_folder = store_folder + "/shard_" + std::to_string(shard_idx);

        fs::path dir_path = new_shard_folder;
        fs::create_directories(dir_path);
        
        uint64_t start_row = shard_idx * rows_per_shard_new;
        uint64_t end_row = min(start_row + rows_per_shard_new, total_vectors_new);

        #pragma omp critical
        {
            std::cout<<"Writing updated matrix in "<<new_shard_folder<<std::endl;
            std::cout<<start_row<<" "<<end_row<<std::endl;
        }

        if(start_row < end_row){
            pc_mat::update_matrix_for_shard(matrix_folder, new_shard_folder, start_row, end_row, acc_vec, new_index_to_prev_index_vec, total_vectors_prev, num_shards);
        }

        
        auto shard_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = shard_end - shard_start;
        auto shard_time = get_time_unit(elapsed.count());
        #pragma omp critical
        {
            std::cout << "Shard " << shard_idx
                      << " completed in "
                      << std::fixed << std::setprecision(2)
                      << shard_time.first << " "
                      << shard_time.second << '\n';
        }
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_total - start_total;
    auto time_unit = get_time_unit(elapsed.count());

    std::cout << "\nAll shards completed in " << std::fixed << std::setprecision(2) 
              << time_unit.first << "\t" << time_unit.second << "\n" << std::endl;
}


std::string get_file_extension(std::string filename){
    size_t dot_pos = filename.find_last_of(".");

    if (dot_pos != std::string::npos) {
        return filename.substr(dot_pos + 1);
    }
    return "";
}


int main(int argc, char* argv[]) {

    // Command line arguments
    string matrix_folder, db_folder, filtered_matrix_folder, acc_db_folder;
    string query_file;
    std::string row_file, col_file;
    // string neighbor_fn = "neighbors.txt";
    uint32_t top_n = 10, batch_size = 1000;
    uint64_t num_acc = 10;
    double filter = 0;
    int n_threads = 1;
    vector<string> query_ids_str;
    bool read_from_stdin = false;
    bool show_help = false;
    bool write_to_file = false;
    std::string out_fn = "out.txt";
    bool print_to_screen = false;
    bool show_all_neighbors = false;
    
    bool use_query_file = false;
    bool use_query_ids = false;
    bool use_row_col_files = false;
    bool use_filter = false;
    bool update_matrix = false;
    bool save_neighbors = false;
    bool only_top_neighbors = false;
    bool only_topn_wfil = false;

    // Change this whenever the file encoding is modified
    const int encoding_version = 1;

    auto cli = (
        clipp::required("--matrix") & clipp::value("folder", matrix_folder),
        clipp::required("--db") & clipp::value("folder", db_folder),
        (
            (clipp::option("--query_file").set(use_query_file) & clipp::value("file", query_file)) |
            (clipp::option("--query_ids").set(use_query_ids) & clipp::values("ids", query_ids_str)) |
            (
            clipp::option("--row_file").set(use_row_col_files) & clipp::value("row", row_file) &
            clipp::option("--col_file") & clipp::value("col", col_file)
            ) |
            (
                clipp::option("--filter").set(use_filter) & clipp::value("double", filter) & 
                clipp::option("--out") & clipp::value("folder", filtered_matrix_folder)

            ) |
            (
                clipp::option("--update").set(update_matrix) & clipp::value("folder", acc_db_folder) & 
                clipp::option("--out") & clipp::value("folder", filtered_matrix_folder)

            ) |
            (
                clipp::option("--nei").set(save_neighbors)
            ) |
            (
                clipp::option("--only").set(only_top_neighbors) & clipp::value("uint64_t", num_acc) & 
                clipp::option("--out") & clipp::value("folder", filtered_matrix_folder)
            ) |
            (
                clipp::option("--nei_fil").set(only_topn_wfil) & clipp::value("uint64_t", num_acc) & clipp::value("double", filter) & 
                clipp::option("--out") & clipp::value("folder", filtered_matrix_folder)
            )

            // | clipp::option("--stdin").set(read_from_stdin)
        ),
        clipp::option("--top") & clipp::value("int", top_n),
        clipp::option("--thread") & clipp::value("int", n_threads),
        clipp::option("--batch_size") & clipp::value("int", batch_size),
        clipp::option("--write_to_file").set(write_to_file) & clipp::value("file", out_fn),
        clipp::option("--show_all").set(show_all_neighbors),
        clipp::option("--print").set(print_to_screen),
        clipp::option("--help").set(show_help)
    );

    if (!clipp::parse(argc, argv, cli) || show_help) {
        cout << "Query Pairwise Comparison Matrix\n\n";
        cout << "Usage:\n" << clipp::usage_lines(cli, argv[0]) << "\n\n";
        cout << "Options:\n";
        cout << "  --matrix        : Folder containing the pairwise matrix files [Required]\n";
        cout << "  --db            : Folder containing the matrix meta data [Required]\n";
        cout << "  --query_file    : File containing query IDs (one per line)\n";
        cout << "  --query_ids     : Query IDs as command line arguments (identifiers separated by space)\n";
        cout << "  --row_file      : File containing query row IDs (one per line)\n";
        cout << "  --col_file      : File containing query col IDs (one per line)\n";
        cout << "  --filter        : Filter values below threshold from matrix\n";
        cout << "  --update        : Update matrix from the provided accession db folder\n";
        cout << "  --out           : Output folder for the filtered matrix\n";
        cout << "  --top           : Number of top jaccard values to show [default 10]\n";
        cout << "  --batch_size    : Number of queries to process per batch [default 1000]\n";
        cout << "  --thread        : Number of threads to use [default 1]\n";
        cout << "  --write_to_file : Where to save the output. Expected format: \n"
             << "                    - *.csv/*tsv/*txt for regular query.\n"     
             << "                    - *.csv/*.tsv/*.npy/*npz/*h5 for row-col query.\n";
        cout << "  --show_all      : Whether to show all neighbors instead of top N\n";
        cout << "  --print         : Whether to print the outputs to screen\n";
        cout << "  --help          : Show this help message\n\n";
        return show_help ? 0 : 1;
    }

    if (matrix_folder.empty()) {
        show_error_and_exit("Error: matrix folder is required.");
    }
    if(!use_query_file && !use_query_ids && !use_row_col_files && !use_filter && !update_matrix && !save_neighbors && !only_top_neighbors && !only_topn_wfil){
        show_error_and_exit("No specific action is given.");
    }

    if (!fs::exists(matrix_folder)) {
        show_error_and_exit("Error: Matrix folder does not exist.");
    }

    // Ensure matrix_folder ends with '/'
    if (!matrix_folder.empty() && matrix_folder.back() != '/' && matrix_folder.back() != '\\') {
        matrix_folder += '/';
    }

    if (!db_folder.empty() && db_folder.back() != '/' && db_folder.back() != '\\') {
        db_folder += '/';
    }

    if(write_to_file && out_fn.empty()){
        show_error_and_exit("No output filename provided.");
    }

    if(use_filter && filtered_matrix_folder.empty()){
        show_error_and_exit("No output folder provided.");
    }

    if(update_matrix && filtered_matrix_folder.empty()){
        show_error_and_exit("No output folder provided.");
    }

    if(!write_to_file) print_to_screen = true;

    string version_file = db_folder + "version.txt";

    ifstream enc_in(version_file);
    int current_version = -1;
    enc_in >> current_version;
    enc_in.close();

    if(current_version != encoding_version){
        show_error_and_exit("Current matrix version is not supported.\nStored matrix version: "
            +to_string(current_version)+"\nCurrent decoder version: "+to_string(encoding_version));
    }

    /**
     * For nearest neighbor queries, all the queries inside the same batch are executed sequentiallyly inside a single thread.
     * Different batches are executed in parallel across multiple threads.
     */

    if(use_query_file || use_query_ids){
        std::string file_extension = get_file_extension(out_fn);
        if(write_to_file){
            if(file_extension != "csv" && file_extension != "tsv" && file_extension != "txt"){
                show_error_and_exit("Output file extension is: "+file_extension+". Expected: csv, tsv or txt.");
            }
        }
        
        std::string sep = file_extension == "csv" ? "," : "\t";
        query_nearest_neighbors(matrix_folder, db_folder, query_file, query_ids_str, 
            write_to_file, show_all_neighbors, top_n, batch_size, out_fn, sep, print_to_screen, n_threads);
    }
    else if(use_row_col_files){
        if(row_file.empty() || col_file.empty()){
            show_error_and_exit("Either row or col file is not specified.");
        }
        std::string file_extension = get_file_extension(out_fn);
        if(write_to_file){
            if(file_extension != "csv" && file_extension != "tsv" && file_extension != "npy" && file_extension != "npz" && file_extension != "h5"){
                show_error_and_exit("Output file extension is: "+file_extension+". Expected: csv, tsv, npy, npz or h5.");
            }
        }
        std::string sep = "-1";
        if(file_extension == "csv" || file_extension == "tsv"){
            sep = file_extension == "csv" ? "," : "\t";
        }

        query_sliced_matrix(matrix_folder, db_folder, row_file, col_file, write_to_file, out_fn, batch_size, print_to_screen, sep, file_extension, n_threads);
    }
    else if(use_filter){
        filter_matrix(matrix_folder, db_folder, filtered_matrix_folder, filter, n_threads);
    }
    else if(update_matrix){
        update_matrix_from_list(matrix_folder, db_folder, filtered_matrix_folder, acc_db_folder, n_threads);
    }
    else if(save_neighbors){
        write_neighbor_from_matrix(matrix_folder, db_folder, n_threads);
    }
    else if(only_top_neighbors){
        store_only_top_n_matrix(matrix_folder, db_folder, filtered_matrix_folder, num_acc, n_threads);
    }
     else if(only_topn_wfil){
        store_only_top_n_wfil_matrix(matrix_folder, db_folder, filtered_matrix_folder, num_acc, filter, n_threads);
    }
    else{
        std::cerr<<"No query types specified. Aborting...\n";
        exit(1);
    }
    
    return 0;
}
