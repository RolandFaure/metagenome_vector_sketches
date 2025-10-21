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

include_directories(${CMAKE_SOURCE_DIR}/include)

# ==== Bits library ====
add_subdirectory(bits)

# ==== Executables ====
add_executable(project_everything
    src/project_everything.cpp
    src/random_projection.cpp
)
target_include_directories(project_everything PRIVATE ${EIGEN_INCLUDE_DIR})

add_executable(standalone_projection
    src/standalone_projection.cpp
    src/random_projection.cpp
)
target_include_directories(standalone_projection PRIVATE ${EIGEN_INCLUDE_DIR})

add_executable(pairwise_comp_optimized
    src/pairwise_comp_optimized.cpp
)
target_include_directories(pairwise_comp_optimized PRIVATE ${EIGEN_INCLUDE_DIR})

add_executable(query_ava_matrix
    src/query_ava_matrix.cpp
)
target_include_directories(query_ava_matrix PRIVATE ${EIGEN_INCLUDE_DIR})

add_library(pc_mat_lib STATIC src/read_pc_mat.cpp)
set_target_properties(pc_mat_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pc_mat_lib PUBLIC include)


add_library(pc_mat_cmp_lib STATIC src/read_pc_mat_cmp.cpp)
set_target_properties(pc_mat_cmp_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pc_mat_cmp_lib PUBLIC 
    ${CMAKE_SOURCE_DIR}/bits/include/
    ${CMAKE_SOURCE_DIR}/bits/external/essentials/include/
)

add_executable(query_pc_mat
    src/query_pc_mat.cpp
)
target_link_libraries(query_pc_mat PRIVATE pc_mat_cmp_lib)

# ==== Link libraries ====
# Only pairwise_comp_optimized depends on the BITS library
target_link_libraries(pairwise_comp_optimized PRIVATE BITS)

# ==== Optional Clean target ====
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E rm -f project_everything standalone_projection pairwise_comp_optimized query_ava_matrix
)


# ==== Python bindings ====
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)  # prefer installed pybind11 config
# If the above fails, install pybind11 or use the pybind11 header-only fallback.

pybind11_add_module(read_pc_mat_module src/bindings.cpp)
target_link_libraries(read_pc_mat_module PRIVATE pc_mat_lib)

set_target_properties(read_pc_mat_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src
)

# Optionally set optimization flags
target_compile_options(read_pc_mat_module PRIVATE -O3 -march=native)
