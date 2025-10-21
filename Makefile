# Makefile for project_everything.cpp, standalone_projection.cpp, and optimized pairwise comparison

CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 -I/usr/include/eigen3 -fopenmp -march=native -ffast-math

TARGETS = project_everything standalone_projection pairwise_comp_optimized query_ava_matrix

all: $(TARGETS)

project_everything: project_everything.cpp random_projection.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o project_everything project_everything.cpp random_projection.cpp

standalone_projection: standalone_projection.cpp random_projection.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o standalone_projection standalone_projection.cpp random_projection.cpp

pairwise_comp_optimized: pairwise_comp_optimized.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o pairwise_comp_optimized pairwise_comp_optimized.cpp

query_ava_matrix: query_ava_matrix.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o query_ava_matrix query_ava_matrix.cpp

# convert_to_zarr: convert_to_zarr.cpp clipp.h
# 	$(CXX) -O3 -Wall -std=c++20 -I/usr/include/eigen3 -fopenmp -march=native -ffast-math -Iinclude/Eigen -I$(CONDA_PREFIX)/include -o convert_to_zarr convert_to_zarr.cpp -L$(CONDA_PREFIX)/lib -lhdf5 -lzstd -pthread

clean:
	rm -f $(TARGETS)