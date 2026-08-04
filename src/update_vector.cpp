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

using namespace std;
vector<std::pair<string, double>> get_sample_norm_vec(std::ifstream& norms_in){
    // vector<double> all_norms;
    vector<std::pair<std::string, double>> sample_norm_vec;
    string line;
    while (getline(norms_in, line)) {
        std::stringstream ss(line);
        string sample;
        double value;
        ss >>sample >> value;
        sample_norm_vec.emplace_back(std::make_pair(sample, value));
    }
    return sample_norm_vec;
}

void add_human_data(const std::string &db_folder, size_t dimension,
        std::ofstream& vec_out, std::ofstream& meta_out){
    
    std::ifstream human_vec_in (db_folder + "human_vectors.bin", std::ios::binary);
    if (!human_vec_in) {
        std::cerr << "Failed to open human_vectors.bin\n";
        return;
    }
    vector<int32_t> buffer(dimension);
    const size_t bytes_per_vector = dimension * sizeof(int32_t);
    human_vec_in.read(reinterpret_cast<char*>(buffer.data()), bytes_per_vector);
    
    if (!human_vec_in) {
        std::cerr << "Failed to read human vector\n";
        return;
    }

    vector<int16_t> updated_vector(dimension);
    for(size_t i=0; i<buffer.size(); i++){
        if(buffer[i] > (int)std::numeric_limits<std::int16_t>::max() || 
                buffer[i] < (int)std::numeric_limits<std::int16_t>::min() ){
            std::cout<<"Human: "<<buffer[i]<<std::endl;
        }
        updated_vector[i] = buffer[i];
    }

    uint64_t write_bytes_per_vector = dimension * sizeof(int16_t);
    // vec_out.write(reinterpret_cast<char*>(buffer.data()), bytes_per_vector);
    vec_out.write(reinterpret_cast<char*>(updated_vector.data()), write_bytes_per_vector);
    
    if (!vec_out) {
        std::cerr << "Human vec Write error\n";
        return;
    }
    meta_out << "human_genome_sketches_merged 1960.17\n";
    if (!meta_out) {
        std::cerr << "Human meta Write error\n";
        return;
    }
}

void filter_and_write(
    const std::vector<std::pair<std::string, double>>& sample_norm_vec,
    const std::string& db_folder,
    const size_t dimension,
    const std::unordered_set<std::string> wgs_set
)
{
    const size_t block_size = 2048;               // vectors per block
    const size_t bytes_per_vector = dimension * sizeof(int32_t);

    std::ifstream vec_in(db_folder + "vectors.bin", std::ios::binary);
    if (!vec_in) {
        std::cerr << "Failed to open input binary file\n";
        return;
    }

    std::ofstream vec_out(db_folder + "non_wgs_filtered_vectors.bin", std::ios::binary);
    std::ofstream meta_out(db_folder + "non_wgs_filtered_sample_norm.txt");

    if (!vec_out || !meta_out) {
        std::cerr << "Failed to open output files\n";
        return;
    }

    size_t total_vectors = sample_norm_vec.size();
    std::ofstream log_out("log.txt");

    for (size_t begin = 0; begin < total_vectors; begin += block_size) {
        std::cout<<"Processing block: "<<begin<<"\n";

        size_t end = std::min(begin + block_size, total_vectors);
        size_t num_vectors = end - begin;

        // contiguous block buffer
        std::vector<int32_t> buffer(num_vectors * dimension);

        vec_in.read(reinterpret_cast<char*>(buffer.data()),
                    num_vectors * bytes_per_vector);

        if (!vec_in) {
            std::cerr << "Error reading block\n";
            return;
        }

        // ---- Filter directly from buffer ----
        for (size_t i = 0; i < num_vectors; ++i) {

            size_t global_i = begin + i;

            if (sample_norm_vec[global_i].second == 0)
                continue;
            if(wgs_set.count(sample_norm_vec[global_i].first) != 0) continue;
            if(sample_norm_vec[global_i].first == "human_genome_sketches_merged") continue;

            // pointer to this vector inside the block
            int32_t* vec_ptr = buffer.data() + i * dimension;
            vector<int16_t> updated_vector(dimension);
            for(size_t i=0; i<dimension; i++){
                if(vec_ptr[i] > (int)std::numeric_limits<std::int16_t>::max() || 
                    vec_ptr[i] < (int)std::numeric_limits<std::int16_t>::min()){
                    std::cout<<"FLAG: "<<global_i<<" "<<i<<" "<<vec_ptr[i]<<" "<<sample_norm_vec[global_i].first << " "
                     << sample_norm_vec[global_i].second
                    <<std::endl;
                }
                updated_vector[i] = vec_ptr[i];
            }
            

            // write directly
            // vec_out.write(reinterpret_cast<char*>(vec_ptr),
            //               bytes_per_vector);
            uint64_t write_bytes_per_vector = dimension * sizeof(int16_t);
            vec_out.write(reinterpret_cast<char*>(updated_vector.data()),
                          write_bytes_per_vector);

            meta_out << sample_norm_vec[global_i].first << " "
                     << sample_norm_vec[global_i].second << "\n";
        }
    }
    // add_human_data(db_folder, dimension, vec_out, meta_out);
    
    vec_in.close();
    vec_out.close();
    meta_out.close();
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

void read_and_filter_vec(std::string &db_folder, std::string &wgs_file){
    int32_t dimension = 2048;

    std::ifstream norm_in(db_folder + "vector_norms.txt");
    vector<pair<string, double>> sample_norm_vec = get_sample_norm_vec(norm_in);
    
    std::ifstream wgs_in(wgs_file);
    std::unordered_set<std::string> wgs_set;
    std::string line;
    int count = 0;
    while(getline(wgs_in, line)){
        wgs_set.insert(get_whitespace_removed(line));
        if(count++ < 10) cout<<line<<endl;
    }

    
    filter_and_write(sample_norm_vec, db_folder, dimension, wgs_set);
}


// void read_and_filter_vec(std::string &db_folder){
//     int32_t dimension = 2048;

//     std::ifstream norm_in(db_folder + "vector_norms.txt");

//     vector<pair<string, double>> sample_norm_vec = get_sample_norm_vec(norm_in);
//     filter_and_write(sample_norm_vec, db_folder, dimension);
// }

int main(int argc, char* argv[]) {
    std::string db_folder = argv[1];
    std::string wgs_file = argv[2];
    read_and_filter_vec(db_folder, wgs_file);

    return 0;
}