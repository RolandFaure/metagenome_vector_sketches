cmake_minimum_required(VERSION 3.10)
project(RandomProjectionTools LANGUAGES CXX)

# ==== Compiler and optimization flags ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

# ==== OpenMP ====
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling support")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ==== Eigen ====
find_path(EIGEN_INCLUDE_DIR Eigen/Dense
    PATHS /usr/include/eigen3
          ${CMAKE_SOURCE_DIR}/include/Eigen
)
if(EIGEN_INCLUDE_DIR)
    message(STATUS "Found Eigen in ${EIGEN_INCLUDE_DIR}")
    include_directories(${EIGEN_INCLUDE_DIR})
else()
    message(WARNING "Eigen not found in standard paths; please set EIGEN_INCLUDE_DIR manually")
endif()

# ==== Bits library ====
# Assuming your bits submodule is located at: ${CMAKE_SOURCE_DIR}/bits
# and that it has its own CMakeLists.txt that defines a target called `BITS`
add_subdirectory(bits)

# ==== Executables ====
add_executable(project_everything
    project_everything.cpp
    random_projection.cpp
)

add_executable(standalone_projection
    standalone_projection.cpp
    random_projection.cpp
)

add_executable(pairwise_comp_optimized
    pairwise_comp_optimized.cpp
)

add_executable(query_ava_matrix
    query_ava_matrix.cpp
)

add_executable(ef_test
    tests/ef_test.cpp
)
target_link_libraries(ef_test PRIVATE BITS)

# ==== Include directories ====
target_include_directories(project_everything PRIVATE ${EIGEN_INCLUDE_DIR})
target_include_directories(standalone_projection PRIVATE ${EIGEN_INCLUDE_DIR})
target_include_directories(pairwise_comp_optimized PRIVATE ${EIGEN_INCLUDE_DIR})
target_include_directories(query_ava_matrix PRIVATE ${EIGEN_INCLUDE_DIR})

# ==== Link libraries ====
# Only pairwise_comp_optimized depends on the BITS library
target_link_libraries(pairwise_comp_optimized PRIVATE BITS)

# ==== Optional Clean target ====
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E rm -f project_everything standalone_projection pairwise_comp_optimized query_ava_matrix
)
