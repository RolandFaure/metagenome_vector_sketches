#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <unordered_set>

#include "random_projection.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <hashes_file> <dimension>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int d = std::stoi(argv[2]);

    std::unordered_set<unsigned long int> hashes;
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::unordered_set<unsigned long int> hashes;
        std::istringstream iss(line);
        unsigned long int hash;
        while (iss >> hash) {
            hashes.insert(hash);
        }

        Eigen::VectorXi vec = transform_set_into_vector(hashes, d);

        for (int i = 0; i < vec.size(); ++i) {
            std::cout << static_cast<float>(vec[i]);
            if (i != vec.size() - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}