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

#include "clipp.h"

namespace fs = std::filesystem;
using namespace std;

struct NeighborData {
    vector<int> neighbor_indices;
    vector<int> neighbor_values;
};

// Function to decompress zstd files if they exist and track them
void decompress_zstd_files(const string& folder) {
    string cmd = "cd " + folder + " && zstd -f -d *.zst 2>/dev/null || true";
    system(cmd.c_str());
}

// Function to clean up all decompressed files
void cleanup_decompressed_files(const string& folder) {
    // Remove decompressed .bin and .txt files, keeping only .zst files
    string cmd = "cd " + folder + " && rm -f matrix.bin row_index.txt 2>/dev/null || true";
    system(cmd.c_str());
}

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

// Load row index mapping from row_index.txt in a specific shard
vector<pair<int, int64_t>> load_shard_row_index(const string& shard_folder) {
    vector<pair<int, int64_t>> address_of_rows;
    
    string index_filename = shard_folder + "/row_index.txt";
    ifstream index_file(index_filename);
    
    if (!index_file) {
        cerr << "Error: Could not open " << index_filename << endl;
        return address_of_rows;
    }
    
    string line;
    while (getline(index_file, line)) {
        istringstream iss(line);
        int row;
        int64_t address;
        if (iss >> row >> address) {
            address_of_rows.push_back({row, address});
        }
    }
    
    return address_of_rows;
}

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
) {
    // Map from shard index to vector of (input index, row)
    unordered_map<int, vector<pair<size_t, int>>> shard_to_queries;
    for (size_t i = 0; i < rows.size(); ++i) {
        int shard_idx = get_shard_for_row(rows[i], total_vectors, num_shards);
        shard_to_queries[shard_idx].emplace_back(i, rows[i]);
    }

    vector<NeighborData> results(rows.size());

    for (const auto& [shard_idx, queries] : shard_to_queries) {
        string shard_folder = matrix_folder + "/shard_" + to_string(shard_idx);

        // Decompress files in this shard
        decompress_zstd_files(shard_folder);

        // Load the row index for this shard
        vector<pair<int, int64_t>> address_of_rows = load_shard_row_index(shard_folder);
        if (address_of_rows.empty()) {
            // All queries in this shard will be empty
            for (const auto& [out_idx, _] : queries) {
                results[out_idx] = NeighborData{};
            }
            // Clean up and continue
            cleanup_decompressed_files(shard_folder);
            continue;
        }

        // Get file size to handle the last row
        string bin_filename = shard_folder + "/matrix.bin";
        ifstream bin_file_size(bin_filename, ios::binary);
        if (!bin_file_size) {
            cerr << "Error: Could not open " << bin_filename << endl;
            for (const auto& [out_idx, _] : queries) {
                results[out_idx] = NeighborData{};
            }
            cleanup_decompressed_files(shard_folder);
            continue;
        }
        bin_file_size.seekg(0, ios::end);
        int64_t file_size = bin_file_size.tellg();
        bin_file_size.close();

        // Build a map from row to address index for fast lookup
        unordered_map<int, size_t> row_to_addr_idx;
        for (size_t i = 0; i < address_of_rows.size(); ++i) {
            row_to_addr_idx[address_of_rows[i].first] = i;
        }

        for (const auto& [out_idx, query_row] : queries) {
            NeighborData result;
            auto it = row_to_addr_idx.find(query_row);
            if (it == row_to_addr_idx.end()) {
                results[out_idx] = result;
                continue;
            }
            size_t addr_idx = it->second;
            int64_t row_address = address_of_rows[addr_idx].second;

            int number_of_neighbors = 0;
            if (addr_idx + 1 < address_of_rows.size()) {
                number_of_neighbors = (address_of_rows[addr_idx + 1].second - row_address) / 8;
            } else {
                number_of_neighbors = (file_size - row_address) / 8;
            }
            if (number_of_neighbors <= 0) {
                results[out_idx] = result;
                continue;
            }
            // Read the neighbor data
            ifstream bin_file(bin_filename, ios::binary);
            if (!bin_file) {
                cerr << "Error: Could not open " << bin_filename << " for row " << query_row << endl;
                results[out_idx] = result;
                continue;
            }
            bin_file.seekg(row_address);

            vector<int32_t> neighbor_differences(number_of_neighbors);
            bin_file.read(reinterpret_cast<char*>(neighbor_differences.data()), number_of_neighbors * sizeof(int32_t));
            vector<int32_t> neighbor_values(number_of_neighbors);
            bin_file.read(reinterpret_cast<char*>(neighbor_values.data()), number_of_neighbors * sizeof(int32_t));

            result.neighbor_indices.resize(number_of_neighbors);
            result.neighbor_values.resize(number_of_neighbors);
            
            int current_col = 0;
            for (int i = 0; i < number_of_neighbors; ++i) {
                current_col += neighbor_differences[i];
                result.neighbor_indices[i] = current_col;
                result.neighbor_values[i] = neighbor_values[i];
            }
            results[out_idx] = std::move(result);

        }

        // Clean up decompressed files for this shard
        cleanup_decompressed_files(shard_folder);
    }

    return results;
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
vector<int> read_queries_from_file(const string& filename, const unordered_map<string, int>& id_to_index) {
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

vector<double> compute_closest_neighbor_distance(
    const string& matrix_folder,
    int total_vectors,
    int num_shards,
    vector<string> identifiers
) {
    vector<double> ratios(total_vectors, -1.0);

    // Prepare all indices
    vector<int> all_rows(total_vectors);
    for (int i = 0; i < total_vectors; ++i) {
        all_rows[i] = i;
    }

    // Batch load all neighbors in batches of one shard size
    int rows_per_shard = (total_vectors + num_shards - 1) / num_shards;

    for (int batch_start = 0; batch_start < total_vectors; batch_start += rows_per_shard) {
        int batch_end = min(batch_start + rows_per_shard, total_vectors);
        vector<int> batch_rows;
        for (int i = batch_start; i < batch_end; ++i) {
            batch_rows.push_back(i);
        }
        vector<NeighborData> batch_neighbors = load_neighbors_for_rows(matrix_folder, batch_rows, total_vectors, num_shards);
        for (size_t i = 0; i < batch_neighbors.size(); ++i) {
            const NeighborData& nd = batch_neighbors[i];
            if (nd.neighbor_values.size() < 2) {
                ratios[batch_rows[i]] = -1.0;
                continue;
            }
            vector<int> sorted_indices(nd.neighbor_values.size());
            std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
            sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
                return nd.neighbor_values[a] > nd.neighbor_values[b];
            });
            int best = nd.neighbor_values[sorted_indices[0]];
            int second_best = nd.neighbor_values[sorted_indices[1]];
            if (second_best != 0) {
                ratios[batch_rows[i]] = second_best / static_cast<double>(best);
            } else {
                ratios[batch_rows[i]] = -1.0;
            }
            if (nd.neighbor_indices.size() >= 2) {
                int idx1 = sorted_indices[0];
                int idx2 = sorted_indices[1];
                int neighbor1 = nd.neighbor_indices[idx1];
                int neighbor2 = nd.neighbor_indices[idx2];
                double ratio = (second_best != 0) ? (second_best / static_cast<double>(best)) : -1.0;
                if (ratio != -1){
                    ratios.push_back(ratio);
                    for (auto _ = 0 ; _ < best ; _+=30){
                        cout << ratio << " ";
                    }
                }
            }
        }
        if (ratios.size() > 1000000){
            cout << endl;
            break;
        }
    }

    return ratios;
}

unordered_map<int, vector<int>> get_neighbors_above_threshold(
    const string& matrix_folder,
    int total_vectors,
    int num_shards,
    vector<float> vector_norms,
    double threshold = 0.3
) {
    unordered_map<int, vector<int>> index_to_neighbors;

    // Process in batches per shard for efficiency
    int rows_per_shard = (total_vectors + num_shards - 1) / num_shards;
    for (int batch_start = 0; batch_start < total_vectors; batch_start += rows_per_shard) {
        int batch_end = min(batch_start + rows_per_shard, total_vectors);
        vector<int> batch_rows;
        for (int i = batch_start; i < batch_end; ++i) {
            batch_rows.push_back(i);
        }
        vector<NeighborData> batch_neighbors = load_neighbors_for_rows(matrix_folder, batch_rows, total_vectors, num_shards);
        for (size_t i = 0; i < batch_neighbors.size(); ++i) {
            const NeighborData& nd = batch_neighbors[i];
            vector<int> filtered_neighbors;
            for (size_t j = 0; j < nd.neighbor_indices.size(); ++j) {
                // Assuming neighbor_values are similarity scores as int, convert to double in [0,1]
                double sim = static_cast<double>(nd.neighbor_values[j]) / vector_norms[batch_rows[i]] ;
                if (sim > threshold) {
                    filtered_neighbors.push_back(nd.neighbor_indices[j]);
                }
            }
            if (!filtered_neighbors.empty()) {
                index_to_neighbors[batch_rows[i]] = std::move(filtered_neighbors);
            }
        }
    }
    return index_to_neighbors;
}

int main(int argc, char* argv[]) {

    // Command line arguments
    string matrix_folder;
    string query_file;
    vector<string> query_ids_str;
    bool read_from_stdin = false;
    bool show_help = false;
    
    auto cli = (
        clipp::option("--matrix_folder") & clipp::value("folder", matrix_folder),
        (
            (clipp::option("--query_file") & clipp::value("file", query_file)) |
            (clipp::option("--query_ids") & clipp::values("ids", query_ids_str)) |
            clipp::option("--stdin").set(read_from_stdin)
        ),
        clipp::option("--help").set(show_help)
    );

    if (!clipp::parse(argc, argv, cli) || show_help) {
        cout << "Query Ava Matrix - Find neighbors in pairwise similarity matrix\n\n";
        cout << "Usage:\n" << clipp::usage_lines(cli, argv[0]) << "\n\n";
        cout << "Options:\n";
        cout << "  --matrix_folder  Folder containing the pairwise matrix files\n";
        cout << "  --query_file     File containing query IDs (one per line)\n";
        cout << "  --query_ids      Query IDs as command line arguments (numeric indices or identifiers)\n";
        cout << "  --stdin          Read query IDs from standard input\n";
        cout << "  --help           Show this help message\n\n";
        cout << "Examples:\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_ids 10 25 42\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_ids SRR123456 SRR789012\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_file queries.txt\n";
        cout << "  echo -e \"SRR123456\\n25\\nSRR789012\" | " << argv[0] << " --matrix_folder ./results --stdin\n";
        return show_help ? 0 : 1;
    }

    if (matrix_folder.empty()) {
        cerr << "Error: --matrix_folder is required" << endl;
        return 1;
    }

    if (!fs::exists(matrix_folder)) {
        cerr << "Error: Matrix folder does not exist: " << matrix_folder << endl;
        return 1;
    }

    // Ensure matrix_folder ends with '/'
    if (!matrix_folder.empty() && matrix_folder.back() != '/' && matrix_folder.back() != '\\') {
        matrix_folder += '/';
    }

    // Load vector identifiers and create mapping
    vector<string> identifiers;
    unordered_map<string, int> id_to_index = load_vector_identifiers(matrix_folder, identifiers);

    vector<float> vector_norms;
    load_vector_norms(matrix_folder, vector_norms);
    
    int total_vectors = identifiers.size();
    std::cout<<"Total vectors loaded: " << total_vectors << endl;
    if (total_vectors <= 0) {
        cerr << "Error: Could not determine total number of vectors" << endl;
        return 1;
    }

    // Discover number of shards
    int num_shards = discover_shards(matrix_folder);
    // num_shards = 100;
    // cout << "DEBUG NUM SHASS" << endl;
    if (num_shards <= 0) {
        cerr << "Error: No shard folders found in " << matrix_folder << endl;
        return 1;
    }

    // cout << "Found " << num_shards << " shards with " << total_vectors << " total vectors" << endl;

    // auto ratios = compute_closest_neighbor_distance(matrix_folder, total_vectors, num_shards, identifiers);
    // exit(0);


    // Determine queries
    vector<int> queries;
    
    if (read_from_stdin) {
        queries = read_queries_from_stdin(id_to_index);
    } else if (!query_file.empty()) {
        queries = read_queries_from_file(query_file, id_to_index);
    } else if (!query_ids_str.empty()) {
        // Convert command line query IDs
        for (const string& query_str : query_ids_str) {
            int index = parse_query_to_index(query_str, id_to_index);
            if (index >= 0) {
                queries.push_back(index);
            }
        }
    } else {
        cerr << "Error: No queries specified. Use --query_file, --query_ids, or --stdin" << endl;
        return 1;
    }

    if (queries.empty()) {
        cerr << "Error: No valid queries found" << endl;
        return 1;
    }

    // Query all at once using load_neighbors_for_rows
    vector<NeighborData> all_neighbors = load_neighbors_for_rows(matrix_folder, queries, total_vectors, num_shards);

    for (size_t q = 0; q < queries.size(); ++q) {
        int query_row = queries[q];
        cout << "Query: " << query_row << " (" << identifiers[query_row] << ")" << endl;

        if (query_row < 0 || query_row >= total_vectors) {
            cout << "  Error: Query row " << query_row << " is out of range [0, " << total_vectors << ")" << endl;
            continue;
        }

        const NeighborData& neighbors = all_neighbors[q];

        if (neighbors.neighbor_indices.empty()) {
            cout << "  No neighbors found" << endl;
        } else {
            cout << "  Found " << neighbors.neighbor_indices.size() << " neighbors:" << endl;
            // Pair each neighbor index with its value (intersection size)
            vector<pair<int, int>> neighbor_pairs;
            for (size_t i = 0; i < neighbors.neighbor_indices.size(); ++i) {
                neighbor_pairs.emplace_back(neighbors.neighbor_indices[i], neighbors.neighbor_values[i]);
            }

            // Sort by Jaccard index (intersection / union) in descending order
            // Jaccard = intersection / (|A| + |B| - intersection)
            // |A| = vector_norms[query_row], |B| = vector_norms[neighbor_idx]
            sort(neighbor_pairs.begin(), neighbor_pairs.end(), [&](const pair<int, int>& a, const pair<int, int>& b) {
                int idx_a = a.first, idx_b = b.first;
                float norm_a = vector_norms[query_row]*vector_norms[query_row];
                float norm_ba = vector_norms[idx_a]*vector_norms[idx_a];
                float norm_bb = vector_norms[idx_b]*vector_norms[idx_b];
                float inter_a = a.second;
                float inter_b = b.second;
                double jac_a = inter_a / (norm_a + norm_ba - inter_a);
                double jac_b = inter_b / (norm_a + norm_bb - inter_b);
                return jac_a > jac_b;
            });

            for (const auto& [neighbor_idx, intersection] : neighbor_pairs) {
                string neighbor_id = (neighbor_idx < total_vectors) ? identifiers[neighbor_idx] : "UNKNOWN";
                float norm_a = vector_norms[query_row]*vector_norms[query_row];
                float norm_b = vector_norms[neighbor_idx]*vector_norms[neighbor_idx];
                double jaccard = intersection / (norm_a + norm_b - intersection);
                // if (neighbor_idx == 34){
                if (jaccard > 0.1 && neighbor_idx < 35000){
                    cout << "  " << neighbor_idx << " (" << neighbor_id << ") intersection=" << intersection
                        << " jaccard=" << jaccard  << " size of the datasets= " << norm_a << " " <<norm_b << endl;
                }
            }
        }
        cout << endl;
    }


    return 0;
}
