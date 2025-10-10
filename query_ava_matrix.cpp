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

#include "clipp.h"

namespace fs = std::filesystem;
using namespace std;

struct NeighborData {
    vector<int> neighbor_indices;
    vector<int> neighbor_values;
};

// Global set to track folders where we decompressed files
unordered_set<string> decompressed_folders;

// Function to decompress zstd files if they exist and track them
void decompress_zstd_files(const string& folder) {
    string cmd = "cd " + folder + " && zstd -f -d *.zst 2>/dev/null || true";
    system(cmd.c_str());
    decompressed_folders.insert(folder);
}

// Function to clean up all decompressed files
void cleanup_decompressed_files() {
    for (const string& folder : decompressed_folders) {
        // Remove decompressed .bin and .txt files, keeping only .zst files
        string cmd = "cd " + folder + " && rm -f matrix.bin row_index.txt 2>/dev/null || true";
        system(cmd.c_str());
    }
    decompressed_folders.clear();
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

// Load neighbors for a specific row from its shard
NeighborData load_neighbors_for_row(const string& matrix_folder, int query_row, 
                                   int total_vectors, int num_shards) {
    NeighborData result;
    
    // Determine which shard contains this row
    int shard_idx = get_shard_for_row(query_row, total_vectors, num_shards);

    cout << "will look in shard " << shard_idx << endl;
    
    string shard_folder = matrix_folder + "/shard_" + to_string(shard_idx);
    
    // Decompress files in this shard if needed
    decompress_zstd_files(shard_folder);
    cout << "decompressed" << endl;
    
    // Load the row index for this shard
    vector<pair<int, int64_t>> address_of_rows = load_shard_row_index(shard_folder);
    if (address_of_rows.empty()) {
        return result;
    }

    // Get file size to handle the last row
    string bin_filename = shard_folder + "/matrix.bin";
    ifstream bin_file(bin_filename, ios::binary);
    if (!bin_file) {
        cerr << "Error: Could not open " << bin_filename << endl;
        return result;
    }
    bin_file.seekg(0, ios::end);
    int64_t file_size = bin_file.tellg();
    int64_t row_address ;
    int number_of_neighbors = 0;
    bool next = false;
    bool found_neighbor = false;
    for (pair<int,int64_t> address : address_of_rows){
        if (next && !found_neighbor){
            number_of_neighbors = (address.second - row_address) / 8;
            cout << "next is " << address.second << endl;
            found_neighbor = true;
        }
        if (address.first == query_row){
            row_address = address.second;
            next = true;
        }
    }
    if (next && !found_neighbor){
        number_of_neighbors = (file_size - row_address) / 8;
    }
    
    cout << "address is " << row_address << " with " << number_of_neighbors << " neighbors" << endl;
    
    // Read the neighbor data
    bin_file.seekg(row_address);
    
    // Read neighbor column differences
    vector<int32_t> neighbor_differences(number_of_neighbors);
    for (int i = 0; i < number_of_neighbors; ++i) {
        bin_file.read(reinterpret_cast<char*>(&neighbor_differences[i]), sizeof(int32_t));
    }
    
    // Read neighbor values
    vector<int32_t> neighbor_values(number_of_neighbors);
    for (int i = 0; i < number_of_neighbors; ++i) {
        bin_file.read(reinterpret_cast<char*>(&neighbor_values[i]), sizeof(int32_t));
    }
    
    // Convert differences to actual indices
    result.neighbor_indices.resize(number_of_neighbors);
    result.neighbor_values.resize(number_of_neighbors);
    
    int current_col = 0;
    for (int i = 0; i < number_of_neighbors; ++i) {
        current_col += neighbor_differences[i];
        result.neighbor_indices[i] = current_col;
        result.neighbor_values[i] = neighbor_values[i];
    }
    
    return result;
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
    
    int total_vectors = identifiers.size();
    if (total_vectors <= 0) {
        cerr << "Error: Could not determine total number of vectors" << endl;
        return 1;
    }

    // Discover number of shards
    int num_shards = discover_shards(matrix_folder);
    if (num_shards <= 0) {
        cerr << "Error: No shard folders found in " << matrix_folder << endl;
        return 1;
    }

    cout << "Found " << num_shards << " shards with " << total_vectors << " total vectors" << endl;

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

    // Process each query
    for (int query_row : queries) {
        cout << "Query: " << query_row << " (" << identifiers[query_row] << ")" << endl;
        
        if (query_row < 0 || query_row >= total_vectors) {
            cout << "  Error: Query row " << query_row << " is out of range [0, " << total_vectors << ")" << endl;
            continue;
        }
        
        NeighborData neighbors = load_neighbors_for_row(matrix_folder, query_row, 
                                                       total_vectors, num_shards);
        
        if (neighbors.neighbor_indices.empty()) {
            cout << "  No neighbors found" << endl;
        } else {
            cout << "  Found " << neighbors.neighbor_indices.size() << " neighbors:" << endl;
            for (size_t i = 0; i < neighbors.neighbor_indices.size(); ++i) {
                int neighbor_idx = neighbors.neighbor_indices[i];
                string neighbor_id = (neighbor_idx < total_vectors) ? identifiers[neighbor_idx] : "UNKNOWN";
                if (neighbor_idx < 35000){
                    cout << "  " << neighbor_idx << " (" << neighbor_id << ") " << neighbors.neighbor_values[i] << endl;
                }
            }
        }
        cout << endl;
    }

    // Clean up all decompressed files before exiting
    cleanup_decompressed_files();

    return 0;
}
