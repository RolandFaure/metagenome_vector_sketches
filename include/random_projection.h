#ifndef RP_H
#define RP_H

#include <Eigen/Dense>
#include <unordered_set>


using Eigen::VectorXi;
using Eigen::VectorXf;

VectorXi transform_set_into_vector(const std::unordered_set<unsigned long int> &hashes, int d);

#endif

