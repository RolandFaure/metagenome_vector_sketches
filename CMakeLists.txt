cmake_minimum_required(VERSION 3.10)
project(DotHash LANGUAGES CXX)

# ==== Compiler and optimization flags ====
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -DNDEBUG")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")

# ==== OpenMP ====
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling support")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ==== Eigen ====
find_path(EIGEN_INCLUDE_DIR Eigen/Dense
    PATHS ${CMAKE_SOURCE_DIR}/include/Eigen
)
if(EIGEN_INCLUDE_DIR)
    message(STATUS "Found Eigen in ${EIGEN_INCLUDE_DIR}")
    include_directories(${EIGEN_INCLUDE_DIR})
else()
    message(WARNING "Eigen not found in standard paths; please set EIGEN_INCLUDE_DIR manually")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# ==== Bits library ====
add_subdirectory(bits)

find_package(ZLIB REQUIRED)

add_library(cnpy ${CMAKE_SOURCE_DIR}/include/cnpy/cnpy.cpp)
target_include_directories(cnpy PUBLIC 
    ${CMAKE_SOURCE_DIR}/include/cnpy/
)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

# ===== HDF5 ======
add_subdirectory(external/HighFive)

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
    src/pairwise_comp_optimized_16bits.cpp
)
target_link_libraries(pairwise_comp_optimized PRIVATE 
    BITS
)

add_executable(update_vec
    src/update_vector.cpp
)


add_library(pc_mat_cmp_lib STATIC src/read_pc_mat_cmp.cpp)
set_target_properties(pc_mat_cmp_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pc_mat_cmp_lib PUBLIC 
    ${CMAKE_SOURCE_DIR}/bits/include/
    ${CMAKE_SOURCE_DIR}/bits/external/essentials/include/
)
target_link_libraries(pc_mat_cmp_lib PUBLIC 
    BITS
)

# add_executable(query_pc_mat
#     src/query_pc_mat.cpp
# )
# target_link_libraries(query_pc_mat PRIVATE pc_mat_lib cnpy)


add_executable(query_pc_mat
    src/query_pc_mat.cpp
)
target_link_libraries(query_pc_mat PRIVATE pc_mat_cmp_lib cnpy HighFive)


# ==== Optional Clean target ====
add_custom_target(clean-all
    COMMAND ${CMAKE_BUILD_TOOL} clean
    COMMAND ${CMAKE_COMMAND} -E rm -f project_everything standalone_projection pairwise_comp_optimized query_ava_matrix query_pc_mat 
)

