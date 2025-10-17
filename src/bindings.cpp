#include "read_pc_mat.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper to convert a std::vector<T> to a NumPy array with zero-copy
template <typename T>
py::array_t<T> vector_to_numpy(std::vector<T> &vec) {
    // shape and strides for 1D contiguous array
    ssize_t n = static_cast<ssize_t>(vec.size());
    std::vector<ssize_t> shape = { n };
    std::vector<ssize_t> strides = { static_cast<ssize_t>(sizeof(T)) };

    // Create a capsule that will free the vector when Python garbage collects the array.
    // We allocate the vector on the heap and transfer ownership to the capsule.
    // Note: we move the vector into a heap-allocated vector to manage lifetime.
    auto *heap_vec = new std::vector<T>(std::move(vec));

    // Create capsule that will delete the heap_vec when array is destroyed.
    py::capsule free_when_done(heap_vec, [](void *p) {
        delete static_cast<std::vector<T>*>(p);
    });

    // Build py::array that uses heap_vec->data()
    return py::array_t<T>(
        shape,
        strides,
        heap_vec->data(),
        free_when_done
    );
}

py::list vector_to_pylist(const std::vector<std::string> &vec) {
    py::list pylist;
    for (const auto &s : vec) {
        pylist.append(s);
    }
    return pylist;
}


// Binding function exposed to Python
py::list query_py(std::string matrix_folder, std::string query_file) {
    std::vector<pc_mat::Result> results = pc_mat::query(matrix_folder, query_file);
    py::list all_results;
    for (const auto &res : results) {
        py::dict res_dict;
        res_dict["id"] = res.self_id;
        res_dict["neighbor_ids"] = vector_to_pylist(res.neighbor_ids);
        res_dict["jaccard_similarities"] = vector_to_numpy(const_cast<std::vector<float>&>(res.jaccard_similarities));
        all_results.append(res_dict);
    }
    return all_results;
}

PYBIND11_MODULE(read_pc_mat_module, m) {
    m.doc() = "Module for querying AVA matrix";
    m.def("query", &query_py, py::arg("matrix_folder"), py::arg("query_file"),
          "Compute neighbors for queries in the given matrix folder and query file; returns a list of dictionaries with neighbor IDs and Jaccard similarities.");
}