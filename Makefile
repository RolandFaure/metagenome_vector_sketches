# Makefile for project_everything.cpp, standalone_projection.cpp, and optimized pairwise comparison

CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 -I/usr/include/eigen3 -fopenmp -march=native -ffast-math

TARGETS = compute_approximate_matrix

all: $(TARGETS)

compute_exact_matrix: compute_exact_matrix.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o compute_exact_matrix compute_exact_matrix.cpp

compute_approximate_matrix: compute_approximate_matrix.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o compute_approximate_matrix compute_approximate_matrix.cpp

# convert_to_zarr: convert_to_zarr.cpp clipp.h
# 	$(CXX) -O3 -Wall -std=c++20 -I/usr/include/eigen3 -fopenmp -march=native -ffast-math -Iinclude/Eigen -I$(CONDA_PREFIX)/include -o convert_to_zarr convert_to_zarr.cpp -L$(CONDA_PREFIX)/lib -lhdf5 -lzstd -pthread

clean:
	rm -f $(TARGETS)