#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <cmath>
#include <limits>
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


// Convert function: Load all signatures and write hashes to a plain text file
void convert(const std::string& folder_name, const std::string& output_file, int num_threads) {
    omp_set_num_threads(num_threads);

    // Timing start
    auto start = std::chrono::high_resolution_clock::now();

    // Collect all signature file paths first
    std::vector<std::string> sig_files;
    for (const auto& entry : fs::directory_iterator(folder_name)) {
        sig_files.push_back(entry.path().string());
    }

    // Open output file
    std::ofstream hash_out(output_file);
    if (!hash_out) {
        std::cerr << "Error opening " << output_file << " for writing." << std::endl;
        return;
    }

    // Process each signature file and write hashes
    std::vector<std::pair<std::string, std::unordered_set<unsigned long int>>> temp_results(sig_files.size());

    // Parallel processing with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < sig_files.size(); ++i) {
        std::unordered_set<unsigned long int> hashes;
        load_signatures(sig_files[i], hashes, omp_get_thread_num());
        
        // Extract the base name (e.g., DRR111514) from the path
        std::string stem = fs::path(sig_files[i]).stem().string();
        std::string base_name = stem.substr(0, stem.find('.'));
        
        temp_results[i] = {base_name, std::move(hashes)};
        
        #pragma omp critical
        {
            cout << "Processed " << sig_files[i] << ", hashes size " << temp_results[i].second.size() << ", file number " << i << endl;
        }
    }

    // Write all results to file
    for (const auto& result : temp_results) {
        hash_out << result.first << ":";
        for (const auto& h : result.second) {
            hash_out << " " << h;
        }
        hash_out << "\n";
    }
    hash_out.close();

    // Timing end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Time to convert all signatures: " << elapsed.count() << " seconds" << endl;
}

// Sketch function: Read hashes from plain text file and create vector sketches
void sketch(const std::string& hash_file, const std::string& index_folder, int dimension, bool use_int16) {
    if (index_folder[index_folder.size()-1] != '/'){
        const_cast<std::string&>(index_folder) += '/';
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

    // Timing start
    auto start = std::chrono::high_resolution_clock::now();

    // Read hashes from file
    std::ifstream hash_in(hash_file);
    if (!hash_in) {
        std::cerr << "Error opening " << hash_file << " for reading." << std::endl;
        return;
    }

    std::vector<std::pair<std::string, std::unordered_set<unsigned long int>>> all_hashes;
    std::string line;
    while (std::getline(hash_in, line)) {
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;
        
        std::string name = line.substr(0, colon_pos);
        std::unordered_set<unsigned long int> hashes;
        
        // Parse hashes
        std::istringstream iss(line.substr(colon_pos + 1));
        unsigned long int hash;
        while (iss >> hash) {
            hashes.insert(hash);
        }
        
        all_hashes.emplace_back(name, std::move(hashes));
    }
    hash_in.close();

    cout << "Loaded " << all_hashes.size() << " hash sets from " << hash_file << endl;

    // Project all hash sets to vectors
    std::vector<std::pair<int, VectorXi>> all_projected_vectors(all_hashes.size());
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < all_hashes.size(); ++i) {
        const auto& hashes = all_hashes[i].second;
        all_projected_vectors[i] = {static_cast<int>(hashes.size()), transform_set_into_vector(hashes, dimension)};
        
        #pragma omp critical
        {
            cout << "Projected " << all_hashes[i].first << ", vector dimension " << dimension << ", index " << i << endl;
        }
    }

    // Timing end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Time to compute all projected vectors: " << elapsed.count() << " seconds" << endl;

    // Output norms and names to a text file, and all vectors as byte-packed int32/int16 to a binary file
    std::ofstream norm_out(index_folder + "vector_norms.txt");
    std::ofstream dim_out(index_folder + "dimension.txt");
    std::ofstream dtype_out(index_folder + "dtype.txt");
    std::ofstream bin_out(index_folder + "vectors.bin", std::ios::binary);
    
    if (!norm_out) {
        std::cerr << "Error opening vector_norms.txt for writing." << std::endl;
    }
    if (!bin_out) {
        std::cerr << "Error opening vectors.bin for writing." << std::endl;
    }
    
    if (norm_out && bin_out && dim_out && dtype_out) {
        dim_out << dimension << "\n";
        dtype_out << (use_int16 ? "int16" : "int32") << "\n";
        
        for (size_t index_of_vector = 0; index_of_vector < all_projected_vectors.size(); ++index_of_vector) {
            const auto& pair = all_projected_vectors[index_of_vector];
            const std::string& base_name = all_hashes[index_of_vector].first;
            
            // Cast vec to VectorXf, divide by sqrt(d), then compute norm
            VectorXi vec = pair.second;
            VectorXf vec_f = pair.second.cast<float>() / std::sqrt(static_cast<float>(dimension));
            double norm = vec_f.norm();
            norm_out << base_name << " " << norm << "\n";
            
            if (use_int16) {
                // Write vector as int16_t, byte-packed, with overflow capping
                constexpr int16_t int16_max = std::numeric_limits<int16_t>::max();
                constexpr int16_t int16_min = std::numeric_limits<int16_t>::min();
                for (int i = 0; i < vec.size(); ++i) {
                    int32_t val32 = static_cast<int32_t>(vec[i]);
                    int16_t val16;
                    if (val32 > int16_max) {
                        val16 = int16_max;
                    } else if (val32 < int16_min) {
                        val16 = int16_min;
                    } else {
                        val16 = static_cast<int16_t>(val32);
                    }
                    bin_out.write(reinterpret_cast<const char*>(&val16), sizeof(int16_t));
                }
            } else {
                // Write vector as int32_t, byte-packed
                for (int i = 0; i < vec.size(); ++i) {
                    int32_t val = static_cast<int32_t>(vec[i]);
                    bin_out.write(reinterpret_cast<const char*>(&val), sizeof(int32_t));
                }
            }
        }
        
        norm_out.close();
        bin_out.close();
        dim_out.close();
        dtype_out.close();
    }
}

int main(int argc, char* argv[]) {
    // CLI with clipp
    bool is_convert = false;
    bool is_sketch = false;
    std::string input_path, output_path;
    int t = 1;
    int d = 2048;
    bool use_int16 = false;

    auto convert_mode = (
        clipp::command("convert").set(is_convert),
        clipp::value("signature_folder", input_path) % "Path to folder containing signature files",
        clipp::value("hash_file", output_path) % "Output hash file path",
        clipp::option("-t", "--threads") & clipp::integer("threads", t) % "Number of threads (default: 1)"
    );

    auto sketch_mode = (
        clipp::command("sketch").set(is_sketch),
        clipp::value("hash_file", input_path) % "Input hash file path",
        clipp::value("index_folder", output_path) % "Output folder for index files",
        clipp::option("-t", "--threads") & clipp::integer("threads", t) % "Number of threads (default: 1)",
        clipp::option("-d", "--dimension") & clipp::integer("dimension", d) % "Vector dimension (default: 2048)",
        clipp::option("--int16").set(use_int16) % "Use int16 instead of int32 for vector storage"
    );

    auto cli = (
        (convert_mode | sketch_mode)
    );

    if (!clipp::parse(argc, argv, cli) || (!is_convert && !is_sketch)) {
        std::cerr << "Usage:\n";
        std::cerr << "  Convert mode:\n";
        std::cerr << "    " << argv[0] << " convert <signature_folder> <hash_file> [-t threads]\n";
        std::cerr << "      signature_folder : Path to folder containing signature files\n";
        std::cerr << "      hash_file        : Output hash file path\n";
        std::cerr << "      -t, --threads    : Number of threads (default: 1)\n\n";
        std::cerr << "  Sketch mode:\n";
        std::cerr << "    " << argv[0] << " sketch <hash_file> <index_folder> [-t threads] [-d dimension] [--int16]\n";
        std::cerr << "      hash_file        : Input hash file path\n";
        std::cerr << "      index_folder     : Output folder for index files\n";
        std::cerr << "      -t, --threads    : Number of threads (default: 1)\n";
        std::cerr << "      -d, --dimension  : Vector dimension (default: 2048)\n";
        std::cerr << "      --int16          : Use int16 instead of int32 for vector storage\n";
        return 1;
    }

    if (is_convert) {
        convert(input_path, output_path, t);
    } else if (is_sketch) {
        sketch(input_path, output_path, d, use_int16);
    }

    return 0;
}