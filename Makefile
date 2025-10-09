# Makefile for project_everything.cpp, standalone_projection.cpp, and optimized pairwise comparison

CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 -I/usr/include/eigen3 -fopenmp -march=native -ffast-math

TARGETS = project_everything standalone_projection pairwise_comp_optimized

all: $(TARGETS)

project_everything: project_everything.cpp random_projection.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o project_everything project_everything.cpp random_projection.cpp

standalone_projection: standalone_projection.cpp random_projection.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o standalone_projection standalone_projection.cpp random_projection.cpp

pairwise_comp_optimized: pairwise_comp_optimized.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o pairwise_comp_optimized pairwise_comp_optimized.cpp

clean:
	rm -f $(TARGETS)