#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <regex>
#include <unordered_set>
#include <numeric>

using namespace std;


namespace pc_mat{
    struct NeighborData {
        vector<int> neighbor_indices;
        vector<int> neighbor_values; // Check: Should this be int64_t?
    };

    // Function to decompress zstd files if they exist and track them
    void decompress_zstd_files(const string& folder);

    // Function to clean up all decompressed files
    void cleanup_decompressed_files(const string& folder);

    // Load vector identifiers and create mapping from identifier to index
    unordered_map<string, int> load_vector_identifiers(const string& matrix_folder, vector<string>& identifiers) ;

    void load_vector_norms(const string& matrix_folder, vector<float>& norms);

    // Get the total number of vectors from vector_norms.txt
    int get_total_vectors(const string& matrix_folder) ;

    // Discover all shard folders and return the number of shards
    int discover_shards(const string& matrix_folder) ;

    // Calculate which shard contains a given row
    int get_shard_for_row(int row, int total_vectors, int num_shards) ;

    // Load row index mapping from row_index.txt in a specific shard
    vector<pair<int, int64_t>> load_shard_row_index(const string& shard_folder) ;

    /*
    * Batch load neighbors for a set of rows.
    * - rows: vector of row indices to query.
    * - Returns: vector<NeighborData> in the same order as input rows.
    * 
    * This function batches queries by shard, decompresses each shard only once,
    * loads all requested rows from that shard, and deletes the uncompressed files after use.
    */
    vector<NeighborData> load_neighbors_for_rows(
        const string& matrix_folder,
        const vector<int>& rows,
        int total_vectors,
        int num_shards
    ) ;


    // Convert query string to index (supports both numeric indices and identifiers)
    int parse_query_to_index(const string& query_str, const unordered_map<string, int>& id_to_index) ;

    // Read queries from file
    vector<int> read_queries_from_file(const string& filename, const unordered_map<string, int>& id_to_index) ;

    // Read queries from stdin
    vector<int> read_queries_from_stdin(const unordered_map<string, int>& id_to_index) ;

    vector<double> compute_closest_neighbor_distance(
        const string& matrix_folder,
        int total_vectors,
        int num_shards,
        vector<string> identifiers
    );

    unordered_map<int, vector<int>> get_neighbors_above_threshold(
        const string& matrix_folder,
        int total_vectors,
        int num_shards,
        vector<float> vector_norms,
        double threshold = 0.3
    );

    struct Result {
        string self_id;
        vector<string> neighbor_ids;
        vector<float> jaccard_similarities;
    };

    vector<Result> query(string matrix_folder, string query_file);
} // namespace pc_mat