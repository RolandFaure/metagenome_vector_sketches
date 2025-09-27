# Makefile for jaccard.cpp and standalone_projection.cpp, both depend on random_projection.cpp

CXX = g++
CXXFLAGS = -O2 -Wall -std=c++17 -I/usr/include/eigen3 -fopenmp

TARGETS = jaccard standalone_projection

all: $(TARGETS)

jaccard: jaccard.cpp random_projection.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o jaccard jaccard.cpp random_projection.cpp

standalone_projection: standalone_projection.cpp random_projection.cpp
	$(CXX) $(CXXFLAGS) -Iinclude/Eigen -o standalone_projection standalone_projection.cpp random_projection.cpp

clean:
	rm -f $(TARGETS)