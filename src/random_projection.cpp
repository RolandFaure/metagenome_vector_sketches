#include "random_projection.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

VectorXi transform_set_into_vector(const std::unordered_set<unsigned long int> &hashes, int d){
    VectorXi vec = VectorXi::Zero(d);
    for (const auto& hash : hashes) {
        for (int i = 0; i < d; i += 64) {
            uint64_t x = static_cast<uint64_t>(hash) + static_cast<uint64_t>(i);
            x += 0x9e3779b97f4a7c15;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
            x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
            x = x ^ (x >> 31);

            for (int n = 0; n < 64 && (i + n) < d; ++n) {
                int projected = 1 - 2 * ((x >> n) & 1);
                vec[i + n] += projected;
            }
        }
    }
    return vec;
}
