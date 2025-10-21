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

#include "z5/factory.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"
#include "xtensor.hpp"
#include "nlohmann/json.hpp"

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
            cleanup_decompressed_files();
            continue;
        }

        // Get file size to handle the last row
        string bin_filename = shard_folder + "/matrix.bin";
        ifstream bin_file(bin_filename, ios::binary);
        if (!bin_file) {
            cerr << "Error: Could not open " << bin_filename << endl;
            for (const auto& [out_idx, _] : queries) {
                results[out_idx] = NeighborData{};
            }
            cleanup_decompressed_files();
            continue;
        }
        bin_file.seekg(0, ios::end);
        int64_t file_size = bin_file.tellg();

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
        cleanup_decompressed_files();
    }

    return results;
}

// Convert matrix to COO sparse Zarr format using z5
void convert_to_zarr(const string& matrix_folder, const string& zarr_path) {
    // Load vector identifiers
    vector<string> identifiers;
    unordered_map<string, int> id_to_index = load_vector_identifiers(matrix_folder, identifiers);
    
    int total_vectors = identifiers.size();
    if (total_vectors <= 0) {
        throw runtime_error("Could not determine total number of vectors");
    }

    // Discover number of shards
    int num_shards = discover_shards(matrix_folder);
    if (num_shards <= 0) {
        throw runtime_error("No shard folders found in " + matrix_folder);
    }

    cout << "Converting " << num_shards << " shards with " << total_vectors << " total vectors to COO sparse Zarr format using z5" << endl;

    // First pass: count total non-zero elements
    int64_t total_nnz = 0;
    cout << "Counting non-zero elements..." << endl;
    
    // Batch process rows for counting non-zero elements
    const int batch_size = 1000;
    for (int start_row = 0; start_row < total_vectors; start_row += batch_size) {
        int end_row = std::min(start_row + batch_size, total_vectors);
        vector<int> batch_rows;
        for (int row = start_row; row < end_row; ++row) {
            batch_rows.push_back(row);
        }
        vector<NeighborData> neighbors_batch = load_neighbors_for_rows(matrix_folder, batch_rows, total_vectors, num_shards);
        for (const auto& neighbors : neighbors_batch) {
            total_nnz += neighbors.neighbor_indices.size();
        }
        cout << "Processed " << end_row << "/" << total_vectors << " rows for counting" << endl;
    }
    
    cout << "Total non-zero elements: " << total_nnz << endl;
    cout << "Sparsity: " << (100.0 * total_nnz) / (static_cast<double>(total_vectors) * total_vectors) << "%" << endl;

    // Remove existing zarr if it exists
    if (fs::exists(zarr_path)) {
        fs::remove_all(zarr_path);
    }

    // Create Zarr group using z5
    z5::filesystem::handle::Group group(zarr_path, z5::FileMode::w);
    group.create();
    
    // Set group attributes
    nlohmann::json group_attrs;
    group_attrs["description"] = "All-vs-all similarity matrix in COO sparse format";
    group_attrs["format"] = "coo_sparse";
    group_attrs["matrix_shape"] = {total_vectors, total_vectors};
    group_attrs["nnz"] = total_nnz;
    group_attrs["total_vectors"] = total_vectors;
    group_attrs["arrays"] = {
        {"row", "Row indices of non-zero elements"},
        {"col", "Column indices of non-zero elements"}, 
        {"data", "Values of non-zero elements"}
    };
    group_attrs["vector_identifiers"] = identifiers;
    
    z5::writeAttributes(group, group_attrs);

    // Determine chunk size for arrays
    std::size_t chunk_size = std::max(1024UL, static_cast<std::size_t>(total_nnz / 100));
    chunk_size = std::min(chunk_size, 1000000UL); // Cap at 1M elements per chunk
    
    cout << "Using chunk size: " << chunk_size << " elements per chunk" << endl;

    // Create COO arrays using z5
    std::vector<std::size_t> shape = {static_cast<std::size_t>(total_nnz)};
    std::vector<std::size_t> chunks = {chunk_size};
    
    // Create arrays with compression using string API
    auto row_array = z5::createDataset(group, "row", "int32", shape, chunks, "gzip");
    auto col_array = z5::createDataset(group, "col", "int32", shape, chunks, "gzip");
    auto data_array = z5::createDataset(group, "data", "int32", shape, chunks, "gzip");
    
    // Collect all COO data
    cout << "Collecting COO sparse data..." << endl;
    
    std::vector<int32_t> all_rows, all_cols, all_data;
    all_rows.reserve(total_nnz);
    all_cols.reserve(total_nnz);
    all_data.reserve(total_nnz);
    
    // Batch process rows for collecting COO data
    for (int start_row = 0; start_row < total_vectors; start_row += num_shards) {
        int end_row = std::min(start_row + num_shards, total_vectors);
        vector<int> batch_rows;
        for (int row = start_row; row < end_row; ++row) {
            batch_rows.push_back(row);
        }
        vector<NeighborData> neighbors_batch = load_neighbors_for_rows(matrix_folder, batch_rows, total_vectors, num_shards);
        for (size_t i = 0; i < neighbors_batch.size(); ++i) {
            int row = start_row + static_cast<int>(i);
            const NeighborData& neighbors = neighbors_batch[i];
            for (size_t j = 0; j < neighbors.neighbor_indices.size(); ++j) {
                all_rows.push_back(row);
                all_cols.push_back(neighbors.neighbor_indices[j]);
                all_data.push_back(neighbors.neighbor_values[j]);
            }
        }
        if (end_row % 1000 == 0 || end_row == total_vectors) {
            cout << "Collected data for " << end_row << "/" << total_vectors << " rows" << endl;
        }
    }
    
    cout << "Writing to Zarr arrays..." << endl;
    
    // Convert to xtensor arrays for z5
    auto row_xt = xt::adapt(all_rows, {static_cast<std::size_t>(total_nnz)});
    auto col_xt = xt::adapt(all_cols, {static_cast<std::size_t>(total_nnz)});
    auto data_xt = xt::adapt(all_data, {static_cast<std::size_t>(total_nnz)});
    
    // Write to zarr using z5
    std::vector<std::size_t> offset = {0};
    z5::multiarray::writeSubarray<int32_t>(row_array, row_xt, offset.begin());
    z5::multiarray::writeSubarray<int32_t>(col_array, col_xt, offset.begin());
    z5::multiarray::writeSubarray<int32_t>(data_array, data_xt, offset.begin());

    cout << "Conversion to COO sparse Zarr format completed: " << zarr_path << endl;
    cout << "Arrays created: row, col, data with " << total_nnz << " elements each" << endl;
}

int main(int argc, char* argv[]) {
    // Command line arguments
    string matrix_folder;
    string zarr_path;
    bool show_help = false;
    
    auto cli = (
        clipp::option("--matrix_folder") & clipp::value("folder", matrix_folder),
        clipp::option("--zarr_path") & clipp::value("path", zarr_path),
        clipp::option("--help").set(show_help)
    );

    if (!clipp::parse(argc, argv, cli) || show_help) {
        cout << "Convert Ava Matrix to Zarr - Convert pairwise similarity matrix to Zarr format\n\n";
        cout << "Usage:\n" << clipp::usage_lines(cli, argv[0]) << "\n\n";
        cout << "Options:\n";
        cout << "  --matrix_folder  Folder containing the pairwise matrix files\n";
        cout << "  --zarr_path      Output path for the Zarr array\n";
        cout << "  --help           Show this help message\n\n";
        cout << "Examples:\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --zarr_path ./similarity_matrix.zarr\n";
        return show_help ? 0 : 1;
    }

    if (matrix_folder.empty() || zarr_path.empty()) {
        cerr << "Error: Both --matrix_folder and --zarr_path are required" << endl;
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

    try {
        convert_to_zarr(matrix_folder, zarr_path);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        cleanup_decompressed_files();
        return 1;
    }

    // Clean up all decompressed files before exiting
    cleanup_decompressed_files();

    return 0;
}
