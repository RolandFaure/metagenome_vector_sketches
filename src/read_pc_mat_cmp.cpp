#include "read_pc_mat.h"
#include "essentials.hpp"
#include "compact_vector.hpp"
#include "elias_fano.hpp"
#include "rice_sequence.hpp"
// #include "streamvbyte.h"
// #include "streamvbytedelta.h"

#include <thread>
#include <chrono>

namespace fs = std::filesystem;

namespace pc_mat {
    // Load vector identifiers and create mapping from identifier to index
    unordered_map<string, int> load_vector_identifiers(const string& matrix_folder, vector<string>& identifiers) {
        unordered_map<string, int> id_to_index;
        
        string norms_file = matrix_folder + "/vector_norms.txt";
        ifstream norms_in(norms_file);
        if (!norms_in) {
            cerr << "Error: Could not open " << norms_file << endl;
            return id_to_index;
        }
        
        string line;
        int index = 0;
        while (getline(norms_in, line)) {
            if (line.empty()) continue;
            
            istringstream iss(line);
            string identifier;
            double norm;
            if (iss >> identifier >> norm) {
                identifiers.push_back(identifier);
                id_to_index[identifier] = index;
                index++;
            }
        }
        
        return id_to_index;
    }

    void load_vector_norms(const string& matrix_folder, vector<float>& norms){
        string norms_file = matrix_folder + "/vector_norms.txt";
        ifstream norms_in(norms_file);
        if (!norms_in) {
            cerr << "Error: Could not open " << norms_file << endl;
            exit(1);
        }
        
        string line;
        while (getline(norms_in, line)) {
            if (line.empty()) continue;
            
            istringstream iss(line);
            string identifier;
            float norm;
            if (iss >> identifier >> norm) {
                norms.push_back(norm);
            }
        }
    }

    // Get the total number of vectors from vector_norms.txt
    int get_total_vectors(const string& matrix_folder) {
        string norms_file = matrix_folder + "/vector_norms.txt";
        ifstream norms_in(norms_file);
        if (!norms_in) {
            cerr << "Error: Could not open " << norms_file << endl;
            return -1;
        }
        
        int count = 0;
        string line;
        while (getline(norms_in, line)) {
            if (!line.empty()) count++;
        }
        return count;
    }

    // Discover all shard folders and return the number of shards
    int discover_shards(const string& matrix_folder) {
        int max_shard = -1;
        
        for (const auto& entry : fs::directory_iterator(matrix_folder)) {
            if (entry.is_directory()) {
                string dirname = entry.path().filename().string();
                regex shard_pattern(R"(shard_(\d+))");
                smatch matches;
                
                if (regex_match(dirname, matches, shard_pattern)) {
                    int shard_num = stoi(matches[1].str());
                    max_shard = max(max_shard, shard_num);
                }
            }
        }
        
        return max_shard + 1; // Number of shards (0-indexed)
    }

    // Calculate which shard contains a given row
    int get_shard_for_row(int row, int total_vectors, int num_shards) {
        int rows_per_shard = (total_vectors + num_shards - 1) / num_shards;
        return row / rows_per_shard;
    }

    
    std::unordered_map<uint32_t, std::pair<uint32_t, uint64_t>> get_shard_row_to_address_map(const string& shard_folder, 
            uint64_t total_vectors, int shard_idx, int num_shards) {
        std::unordered_map<uint32_t, std::pair<uint32_t, uint64_t> > row_to_address_map;
        
        string index_filename = shard_folder + "/row_index.bin";
        ifstream index_file(index_filename, ios::binary);
        
        if (!index_file) {
            cerr << "Error: Could not open " << index_filename << endl;
            return row_to_address_map;
        }
        // bits::compact_vector row_cv;
        bits::compact_vector delta_address_cv;
        // row_cv.load(index_file);
        delta_address_cv.load(index_file);

        uint64_t rows_per_shard = (total_vectors + num_shards - 1) / num_shards;
        uint64_t curr_row = shard_idx * rows_per_shard;
        uint64_t end_row = min(curr_row + rows_per_shard, total_vectors);

        // first position is always 0, rest are delta coded

        // uint32_t curr_row = row_cv.access(0);
        
        row_to_address_map[curr_row] = std::make_pair(0, 0);
        uint32_t prev_row = curr_row;
        
        for (size_t i = 1; i < delta_address_cv.size()+1; ++i) {
            // std::cout<<"Loaded row index "<< i <<": "<< rs_row.access(i)
            //     <<" address: "<< rs_address.access(i) << std::endl;
            // curr_row = row_cv.access(i);
            curr_row++;
            uint64_t curr_add = row_to_address_map[prev_row].second + delta_address_cv.access(i-1);
            row_to_address_map[curr_row] = std::make_pair(i, curr_add);
            prev_row = curr_row;
        }
        
        return row_to_address_map;
    }

    std::vector<Neighbors> load_neighbors_for_rows(
        const string& matrix_folder,
        const vector<int>& rows,
        uint32_t total_vectors,
        int num_shards
    ) {
        // Map from shard index to query index
        // we will process all queries from the same shard together and go the the next shard
        unordered_map<int, std::vector<uint32_t> > shard_to_queries;
        for (uint32_t i = 0; i < rows.size(); ++i) {
            int shard_idx = get_shard_for_row(rows[i], total_vectors, num_shards);
            // int shard_idx = 0;
            shard_to_queries[shard_idx].emplace_back(i);
        }

        // std::vector<NeighborData> results(rows.size());
        std::vector<Neighbors> results(rows.size());
        
        for(const auto &[shard_idx, query_index_vec]: shard_to_queries){
            // std::cout<<"shard: "<<shard_idx<<std::endl;
            
            std::string shard_folder = matrix_folder + "/shard_" + std::to_string(shard_idx);
            // decompress_zstd_files(shard_folder);
            const std::unordered_map<uint32_t, std::pair<uint32_t, uint64_t>>& row_to_indx_add_map = get_shard_row_to_address_map(shard_folder, 
                total_vectors, shard_idx, num_shards);
            assert(!row_to_indx_add_map.empty());
            
            std::string bin_fn = shard_folder + "/matrix.bin";
            std::ifstream bin_in(bin_fn, std::ios::binary);

            std::string ngh_fn = shard_folder + "/neighbor_start.bin";
            std::ifstream ngh_in(ngh_fn, std::ios::binary);
            bits::rice_sequence<> rs_start;
            rs_start.load(ngh_in);
            ngh_in.close();
            // std::cout<<"cv loaded, size: "<<cv_vb.size()<<std::endl;
            for(const uint32_t& query_index: query_index_vec){
                Neighbors result;
                uint32_t curr_row = rows[query_index];
                auto it = row_to_indx_add_map.find(curr_row);
                if (it == row_to_indx_add_map.end()) {
                    results[query_index] = result;
                    continue;
                }
                uint32_t curr_query_build_index = it->second.first;
                uint64_t curr_add = it->second.second;
                // std::cout<<"cqbi: "<<curr_query_build_index<<" crow: "<<curr_row<<" ca: "<<curr_add<<std::endl;
                bin_in.clear();
                bin_in.seekg(curr_add, std::ios::beg);

                bits::compact_vector cv_jc;
                cv_jc.load(bin_in);
                
                bits::rice_sequence<> rs_delta;
                
                if(cv_jc.size() > 1) rs_delta.load(bin_in);
                
                uint64_t number_of_neighbors = cv_jc.size();
                
                
                result.index_jaccard.resize(number_of_neighbors);
                
                result.index_jaccard[0] = std::make_pair(rs_start.access(curr_query_build_index), 
                                                cv_jc.access(0));

                for(int i=1; i<number_of_neighbors; i++){
                    result.index_jaccard[i] = std::make_pair(result.index_jaccard[i-1].first 
                                + rs_delta.access(i-1), cv_jc.access(i));
                    // if(i < 10)
                    //     std::cout<<result.index_jaccard[i].first<<" "<<result.index_jaccard[i].second<<std::endl;
                }
                results[query_index] = std::move(result);
            }
            // cleanup_decompressed_files(shard_folder);
        }
        return results;
    }

    void filter_matrix(std::string shard_folder, uint64_t start_row, uint64_t end_row, double filter){
        const double MULT_CONST = (1ULL << 8) - 1;
        uint32_t threshold = round(filter * MULT_CONST); //everything below threshold will be skipped
        


    }

    // Convert query string to index (supports both numeric indices and identifiers)
    int parse_query_to_index(const string& query_str, const unordered_map<string, int>& id_to_index) {
        // First try to parse as a number
        try {
            int index = stoi(query_str);
            return index;
        } catch (const exception& e) {
            // If parsing as number fails, try to look up as identifier
            auto it = id_to_index.find(query_str);
            if (it != id_to_index.end()) {
                return it->second;
            } else {
                cerr << "Warning: Could not find identifier '" << query_str << "'" << endl;
                return -1; // Invalid index
            }
        }
    }

    // Read queries from file
    vector<int> read_queries_from_file(const string& filename, const unordered_map<string, int>& id_to_index,
            std::vector<std::string>& id_vec) {
        vector<int> queries;
        ifstream file(filename);
        
        if (!file) {
            cerr << "Error: Could not open query file " << filename << endl;
            return queries;
        }
        
        string line;
        while (getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;
            
            // Remove leading/trailing whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            int index = parse_query_to_index(line, id_to_index);
            if (index >= 0) {
                queries.push_back(index);
                id_vec.push_back(line);
            }
            else{
                // id_vec.push_back("UNKNOWN");
            }
        }
        
        return queries;
    }

    // Read queries from stdin
    vector<int> read_queries_from_stdin(const unordered_map<string, int>& id_to_index) {
        vector<int> queries;
        string line;
        
        while (getline(cin, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;
            
            // Remove leading/trailing whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            int index = parse_query_to_index(line, id_to_index);
            if (index >= 0) {
                queries.push_back(index);
            }
        }
        
        return queries;
    }

    vector<Result> query(std::string matrix_folder, vector<int>& queries, 
        std::vector<float>& vector_norms, std::vector<string>& identifiers){
        
        // Discover number of shards
        int num_shards = discover_shards(matrix_folder);
        // num_shards = 100;
        // int num_shards = 1000;
        // cout << "DEBUG NUM SHASS" << endl;
        if (num_shards <= 0) {
            cerr << "Error: No shard folders found in " << matrix_folder << endl;
        }
        uint32_t total_vectors = vector_norms.size();
        // cout << "Found " << num_shards << " shards with " << total_vectors << " total vectors" << endl;

        const double MULT_CONST = (1ULL << 8) - 1;

        // Query all at once using load_neighbors_for_rows
        std::vector<Neighbors> all_neighbors = load_neighbors_for_rows(matrix_folder, queries, total_vectors, num_shards);

        vector<Result> all_results(queries.size());
        for (size_t q = 0; q < queries.size(); ++q) {
            int query_row = queries[q];
            // cout << "Query: " << query_row << " (" << identifiers[query_row] << ")" << endl;

            if (query_row < 0 || query_row >= total_vectors) {
                cout << "  Error: Query row " << query_row << " is out of range [0, " << total_vectors << ")" << endl;
                continue;
            }

            Neighbors& neighbors = all_neighbors[q];


            if (neighbors.index_jaccard.empty()) {
                cout << "  No neighbors found" << endl;
            } 
            else {
                sort(neighbors.index_jaccard.begin(), neighbors.index_jaccard.end(), 
                        [] (const std::pair<uint64_t, uint32_t>& a, const std::pair<uint64_t, uint32_t>& b) {
                    return a.second > b.second;
                });
                Result res;
                res.self_id = identifiers[query_row];
                for(size_t n=0; n<neighbors.index_jaccard.size(); n++){
                    uint64_t neighbor_idx = neighbors.index_jaccard[n].first;
                    uint32_t neighbor_jaccard = neighbors.index_jaccard[n].second;

                    std::string neighbor_id = (neighbor_idx < total_vectors) ? 
                                    identifiers[neighbor_idx] : "UNKNOWN";
                    res.neighbor_ids.push_back(neighbor_id);
                    res.jaccard_similarities.push_back(static_cast<double>(neighbor_jaccard)/MULT_CONST);
                    // if(n < 10)
                    //     std::cout<<neighbor_jaccard<<" "<<static_cast<double>(neighbor_jaccard)/MULT_CONST<<std::endl;
                }
                all_results[q] = std::move(res);
            }
        }
        return all_results;
    }

    vector<std::vector<uint32_t> > load_neighbors_for_slice(
        const string& matrix_folder,
        const vector<int>& rows,
        const vector<int>& cols,
        int total_vectors,
        int num_shards
    )
    {
        //TODO: replace the map with vector(num_shards)
        unordered_map<int, std::vector<uint32_t> > shard_to_queries;
        for (size_t i = 0; i < rows.size(); ++i) {
            int shard_idx = get_shard_for_row(rows[i], total_vectors, num_shards);
            shard_to_queries[shard_idx].emplace_back(i);
        }

        std::vector<std::vector<uint32_t> >results(rows.size());

        for (const auto& [shard_idx, query_index_vec] : shard_to_queries) {
            string shard_folder = matrix_folder + "/shard_" + to_string(shard_idx);

            // Decompress files in this shard
            // decompress_zstd_files(shard_folder);

            // Load the row index for this shard
            const std::unordered_map<uint32_t, std::pair<uint32_t, uint64_t>>& row_to_indx_add_map = get_shard_row_to_address_map(shard_folder, 
                total_vectors, shard_idx, num_shards);
            assert(!row_to_indx_add_map.empty());
            
            
            std::string bin_fn = shard_folder + "/matrix.bin";
            std::ifstream bin_in(bin_fn, std::ios::binary);

            std::string ngh_fn = shard_folder + "/neighbor_start.bin";
            std::ifstream ngh_in(ngh_fn, std::ios::binary);
            bits::rice_sequence<> rs_start;
            rs_start.load(ngh_in);
            ngh_in.close();

            for(const uint32_t& query_index: query_index_vec){
                std::vector<uint32_t> result;
                uint32_t curr_row = rows[query_index];
                auto it = row_to_indx_add_map.find(curr_row);
                if (it == row_to_indx_add_map.end()) {
                    results[query_index] = std::move(result);
                    continue;
                }
                uint32_t curr_query_build_index = it->second.first;
                uint64_t curr_add = it->second.second;
                // std::cout<<"cqbi: "<<curr_query_build_index<<" crow: "<<curr_row<<" ca: "<<curr_add<<std::endl;
                bin_in.clear();
                bin_in.seekg(curr_add, std::ios::beg);

                bits::compact_vector cv_jc;
                cv_jc.load(bin_in);
                
                bits::rice_sequence<> rs_delta;
                
                if(cv_jc.size() > 1) rs_delta.load(bin_in);
                
                uint64_t number_of_neighbors = cv_jc.size();

                unordered_map<int64_t, int64_t> row_to_jaccard_map;
                int64_t neighbor_index = rs_start.access(curr_query_build_index);
                row_to_jaccard_map[neighbor_index] = cv_jc.access(0);
                for(int i=1; i<number_of_neighbors; i++){
                    neighbor_index += rs_delta.access(i-1);
                    row_to_jaccard_map[neighbor_index] = cv_jc.access(i);
                }

                result.reserve(cols.size());
                
                for(int64_t i=0; i<cols.size(); i++){
                    int64_t col_indx = cols[i];
                    auto it = row_to_jaccard_map.find(col_indx);
                    if (it == row_to_jaccard_map.end()) {
                        result.push_back(0);
                    }
                    else{
                        result.push_back(it->second);
                    }
                }
                results[query_index] = std::move(result);
            }
            // cleanup_decompressed_files(shard_folder);
        }

        return results;
    }

    std::vector<std::vector<float> > query_sliced(std::string matrix_folder, std::vector<int32_t>& row_queries_vec, 
        std::vector<int32_t>& col_queries_vec, int32_t total_vectors,
        std::vector<float>& vector_norms
    ){
        
        int num_shards = discover_shards(matrix_folder);
        if (num_shards <= 0) {
            cerr << "Error: No shard folders found in " << matrix_folder << endl;
        }

        const double MULT_CONST = (1ULL << 8) - 1;

        vector<std::vector<uint32_t> > all_neighbors = load_neighbors_for_slice(matrix_folder, row_queries_vec, 
            col_queries_vec, total_vectors, num_shards);

        vector<std::vector<float> > all_results(row_queries_vec.size());

        for (size_t q = 0; q < row_queries_vec.size(); ++q) {
            int query_row = row_queries_vec[q];
            // cout << "Query: " << query_row << " (" << identifiers[query_row] << ")" << endl;

            std::vector<uint32_t>& neighbors = all_neighbors[q];
            std::vector<float> res;
            if(neighbors.empty()){
                for(size_t i=0; i<col_queries_vec.size(); i++) res.push_back(0);
            }
            else{
                assert(neighbors.size() == col_queries_vec.size());
                for(size_t i=0; i<col_queries_vec.size(); i++){
                    res.push_back( static_cast<double>(neighbors[i]) / MULT_CONST);
                }
            }
            all_results[q] = std::move(res);
        }
        return all_results;
    }

} // namespace pc_mat

