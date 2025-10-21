#include <iostream>
#include <fstream>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <omp.h>
#include <zlib.h>

#include "random_projection.h"
#include "clipp.h"

using Eigen::VectorXi;
using Eigen::VectorXf;
using std::string;
using std::cout;
using std::endl;
using std::unordered_set;
using std::vector;
namespace fs = std::filesystem;

// Extract all 31-mers from a fasta file and store in a set
std::unordered_set<std::string> extract_31mers(const std::string& fasta_path) {
    std::unordered_set<std::string> kmers;
    std::ifstream infile(fasta_path);
    if (!infile) {
        std::cerr << "Error opening file: " << fasta_path << std::endl;
        return kmers;
    }
    std::string line, seq;
    auto nb_line = 0;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!seq.empty()) seq.clear();
            continue;
        }
        seq += line;
        // Extract 31-mers from the current sequence
        if (seq.size() >= 31) {
            for (size_t i = 0; i <= seq.size() - 31; ++i) {
                std::string kmer = seq.substr(i, 31);
                std::transform(kmer.begin(), kmer.end(), kmer.begin(), ::toupper);
                if (kmer.find_first_not_of("ACGT") == std::string::npos)
                    kmers.insert(kmer);
            }
        }
        cout << "parsed " << nb_line++ << " lines\r" << std::flush;
    }
    return kmers;
}

// Compute Jaccard distance between two sets
double jaccard_distance(const std::unordered_set<std::string>& set1,
                        const std::unordered_set<std::string>& set2) {
    size_t intersection = 0;
    for (const auto& kmer : set1) {
        if (set2.count(kmer)) ++intersection;
    }
    size_t union_size = set1.size() + set2.size() - intersection;
    if (union_size == 0) return 0.0;
    double jaccard_index = static_cast<double>(intersection) / union_size;
    return 1.0 - jaccard_index;
}


// Helper to read gzipped file into a string
std::string read_gzipped_file(const std::string& gz_path) {
    // Use system gunzip to decompress, read file, then delete
    std::string temp_file = gz_path.substr(0, gz_path.size() - 3); // Remove .gz
    std::string gunzip_cmd = "gunzip -c " + gz_path + " > " + temp_file + " 2>/dev/null";
    int ret = system(gunzip_cmd.c_str());
    if (ret != 0) {
        std::cerr << "Error running gunzip on: " << gz_path << std::endl;
        return "";
    }
    std::ifstream infile(temp_file);
    if (!infile) {
        std::cerr << "Error opening decompressed file: " << temp_file << std::endl;
        std::remove(temp_file.c_str());
        return "";
    }
    std::string contents((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();
    std::remove(temp_file.c_str());
    return contents;
}

void load_signatures(std::string file_name, std::unordered_set<unsigned long int> &hashes, int thread_id){
    // file_name is a zip file containing a "signatures" folder
    // In this folder are gzipped files with JSON arrays containing "mins"
    std::string temp_dir = "/tmp/signature_extract"+std::to_string(thread_id);
    std::string unzip_cmd = "unzip -qq -o " + file_name + " -d " + temp_dir + " 2>/dev/null";
    int ret = system(unzip_cmd.c_str());
    if (ret != 0) {
        std::cerr << "Failed to unzip: " << file_name << std::endl;
        return;
    }
    std::string sig_folder = temp_dir + "/signatures";
    for (const auto& entry : fs::directory_iterator(sig_folder)) {
        if (entry.path().extension() == ".gz") {
            std::string json_str = read_gzipped_file(entry.path().string());
            if (json_str.empty()) continue;
            size_t ksize_pos = json_str.find("\"ksize\"");
            if (ksize_pos == std::string::npos) continue;
            size_t colon_pos = json_str.find(':', ksize_pos);
            if (colon_pos == std::string::npos) continue;
            size_t ksize_end = json_str.find_first_of(",}", colon_pos);
            std::string ksize_str = json_str.substr(colon_pos + 1, ksize_end - colon_pos - 1);
            ksize_str.erase(std::remove_if(ksize_str.begin(), ksize_str.end(), ::isspace), ksize_str.end());
            if (ksize_str != "31") continue;

            // Manually extract the "mins" array from the JSON string
            size_t mins_pos = json_str.find("\"mins\"");
            if (mins_pos == std::string::npos) continue;
            size_t array_start = json_str.find('[', mins_pos);
            size_t array_end = json_str.find(']', array_start);
            if (array_start == std::string::npos || array_end == std::string::npos) continue;
            std::string array_str = json_str.substr(array_start + 1, array_end - array_start - 1);

            // Split by comma and parse each value
            size_t pos = 0;
            while (pos < array_str.size()) {
                // Skip whitespace
                while (pos < array_str.size() && std::isspace(array_str[pos])) ++pos;
                size_t next_comma = array_str.find(',', pos);
                std::string num_str;
                if (next_comma == std::string::npos) {
                    num_str = array_str.substr(pos);
                    pos = array_str.size();
                } else {
                    num_str = array_str.substr(pos, next_comma - pos);
                    pos = next_comma + 1;
                }
                // Remove whitespace
                num_str.erase(std::remove_if(num_str.begin(), num_str.end(), ::isspace), num_str.end());
                if (!num_str.empty()) {
                    try {
                        uint64_t val = std::stoull(num_str);
                        hashes.insert(val);
                    } catch (...) {
                        // Ignore parse errors
                    }
                }
            }
        }
    }
    // Optionally, clean up temp_dir if desired
    std::string cleanup_cmd = "rm -rf " + temp_dir;
    int cleanup_ret = system(cleanup_cmd.c_str());
    if (cleanup_ret != 0) {
        std::cerr << "Failed to clean up temp directory: " << temp_dir << std::endl;
    }

    #pragma omp critical
    {
        // Extract the base name (e.g., DRR111514) from the path
        std::string stem = fs::path(file_name).stem().string();
        std::string base_name = stem.substr(0, stem.find('.'));
        static std::ofstream hash_out("all_hashes.txt", std::ios::app);
        if (hash_out) {
            hash_out << base_name << ":";
            for (const auto& h : hashes) {
                hash_out << " " << h;
            }
            hash_out << "\n";
            hash_out.flush();
        } else {
            std::cerr << "Error opening all_hashes.txt for writing." << std::endl;
        }
    }
}

VectorXi transform_set_into_minHash_vector(const std::unordered_set<unsigned long int> &hashes, int d){
    std::vector<unsigned long int> sorted_hashes(hashes.begin(), hashes.end());
    std::sort(sorted_hashes.begin(), sorted_hashes.end());
    VectorXi vec = VectorXi::Zero(d);
    size_t n = std::min(static_cast<size_t>(d), sorted_hashes.size());
    for (size_t i = 0; i < n; ++i) {
        vec[i] = static_cast<int>(sorted_hashes[i]);
    }
    return vec;
}


int main(int argc, char* argv[]) {
    // CLI with clipp
    int t = 1;
    int d = 2048;
    int strategy = 0;
    std::string folder_name, index_folder;

    auto cli = (
        clipp::value("input_folder", folder_name),
        clipp::value("index_folder", index_folder),
        clipp::option("-t", "--threads") & clipp::integer("threads", t) % "Number of threads (default: 1)",
        clipp::option("-d", "--dimension") & clipp::integer("dimension", d) % "Vector dimension (default: 2048)",
        clipp::option("-s", "--strategy") & clipp::integer("strategy", strategy) % "0=random projections, 1=minHash (default: 0)"
    );

    if (!clipp::parse(argc, argv, cli)) {
        std::cerr << "Usage: " << argv[0] << " <input_folder> <index_folder> [-t threads] [-d dimension] [-s strategy]\n";
        std::cerr << "  strategy: 0=random projections, 1=minHash\n";
        return 1;
    }

    omp_set_num_threads(t);
    if (index_folder[index_folder.size()-1] != '/'){
        index_folder += '/';
    }
    // Ensure index_folder exists and is empty
    if (fs::exists(index_folder)) {
        // Remove all contents if not empty
        for (const auto& entry : fs::directory_iterator(index_folder)) {
            fs::remove_all(entry.path());
        }
    } else {
        // Create the directory if it doesn't exist
        fs::create_directories(index_folder);
    }


    std::vector<std::unordered_set<unsigned long int>> all_hash_sets;
    std::vector<std::pair<int, VectorXi>> all_projected_vectors;
    // std::vector<std::string> folder_names;
    int nb_loads = 0;

    // Timing start
    auto start = std::chrono::high_resolution_clock::now();

    // Collect all signature file paths first
    std::vector<std::string> sig_files;
    for (const auto& entry : fs::directory_iterator(folder_name)) {
        sig_files.push_back(entry.path().string());
    }

    // Prepare storage for results
    std::vector<std::pair<int, VectorXi>> temp_projected_vectors(sig_files.size());

    // Parallel processing with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < sig_files.size(); ++i) {
        std::unordered_set<unsigned long int> hashes;
        load_signatures(sig_files[i], hashes, omp_get_thread_num());
        if (strategy == 0){
            temp_projected_vectors[i] = {static_cast<int>(hashes.size()), transform_set_into_vector(hashes, d)};
        }
        else{
            temp_projected_vectors[i] = {static_cast<int>(hashes.size()), transform_set_into_minHash_vector(hashes, d)};
        }
        #pragma omp critical
        {
            cout << "Processed " << sig_files[i] << ", hashes size " << hashes.size() << ", file number " << i << endl;
        }
    }
 
    // Move results to main vector
    all_projected_vectors = std::move(temp_projected_vectors);

    // Timing end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Time to compute all projected vectors: " << elapsed.count() << " seconds" << endl;

    // Output norms and names to a text file, and all vectors as byte-packed int32 to a binary file
    std::ofstream norm_out(index_folder + "vector_norms.txt");
    std::ofstream dim_out(index_folder + "dimension.txt");
    std::ofstream bin_out(index_folder + "vectors.bin", std::ios::binary);
    if (!norm_out) {
        std::cerr << "Error opening vector_norms.txt for writing." << std::endl;
    }
    if (!bin_out) {
        std::cerr << "Error opening vectors.bin for writing." << std::endl;
    }
    if (norm_out && bin_out && dim_out) {
        dim_out << d << "\n";
        int index_of_vector = 0;
        for (const auto& pair : all_projected_vectors) {
            // Extract the base name (DRR111514) from the path
            std::string stem = fs::path(sig_files[index_of_vector]).stem().string();
            std::string base_name = stem.substr(0, stem.find('.'));
            // Cast vec to VectorXf, divide by sqrt(d), then compute norm
            VectorXi vec = pair.second;
            VectorXf vec_f = pair.second.cast<float>() / std::sqrt(static_cast<float>(d));
            double norm = vec_f.norm();
            norm_out << base_name << " " << norm << "\n";
            // Write vector as int32_t, byte-packed
            for (int i = 0; i < vec.size(); ++i) {
                int32_t val = static_cast<int32_t>(vec[i]);
                bin_out.write(reinterpret_cast<const char*>(&val), sizeof(int32_t));
            }
            index_of_vector++;
        }
        norm_out.close();
        bin_out.close();
        dim_out.close();
    }

    // std::vector<std::pair<double, double>> jaccard_pairs;
    // size_t count_above_01 = 0;
    // for (size_t i = 0; i < all_hash_sets.size(); ++i) {
    //     for (size_t j = i + 1; j < all_hash_sets.size(); ++j) {
    //         size_t intersection = 0;
    //         const auto& set1 = all_hash_sets[i];
    //         const auto& set2 = all_hash_sets[j];
    //         for (const auto& hash : set1) {
    //             if (set2.count(hash)) ++intersection;
    //         }
    //         size_t union_size = set1.size() + set2.size() - intersection;
    //         double jaccard = union_size == 0 ? 0.0 : static_cast<double>(intersection) / union_size;

    //         vector<float> vec1 = all_projected_vectors[i].second;
    //         int size_1 = all_projected_vectors[i].first;
    //         vector<float> vec2 = all_projected_vectors[j].second;
    //         int size_2 = all_projected_vectors[j].first;

    //         // Compute squared Euclidean norm of the difference between the two vectors
    //         double squared_norm = 0.0;
    //         for (int k = 0; k < d; ++k) {
    //             double diff = vec1[k] - vec2[k];
    //             squared_norm += diff * diff;
    //         }

    //         double estimated_jaccard = (size_1 + size_2 - squared_norm) / (size_1 + size_2 + squared_norm);

    //         jaccard_pairs.emplace_back(jaccard, estimated_jaccard);

    //         if (jaccard >= 0.1){
    //             cout << folder_names[i] << " vs " << folder_names[j] << ": " << jaccard
    //                  << " : " << estimated_jaccard << endl;
    //         }
    //         if (jaccard > 0.1) ++count_above_01;
    //     }
    // }

    // // Output each (x, y) pair to points.txt, one per line
    // std::ofstream outfile("points.txt");
    // if (!outfile) {
    //     std::cerr << "Error opening points.txt for writing." << std::endl;
    // } else {
    //     for (const auto& pair : jaccard_pairs) {
    //         if (pair.first >= 0.0){
    //             outfile << pair.first << " " << pair.second << "\n";
    //         }
    //     }
    //     outfile.close();
    // }
    // cout << "Number of pairwise Jaccard distances above 0.1: " << count_above_01 << endl;

    return 0;
}