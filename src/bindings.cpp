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
py::list query_py(std::string matrix_folder, std::string db_folder, std::string query_file) {
    std::vector<string> identifiers;
    unordered_map<string, int> id_to_index = pc_mat::load_vector_identifiers(db_folder, identifiers);
    
    std::vector<int32_t> queries;
    std::vector<std::string> query_ids_str;
    
    queries = pc_mat::read_queries_from_file(query_file, id_to_index, query_ids_str);
    
    // int total_vectors = identifiers.size();

    std::vector<float> vector_norms;
    pc_mat::load_vector_norms(db_folder, vector_norms);

    std::vector<pc_mat::Result> results = pc_mat::query(matrix_folder, queries, vector_norms, identifiers);
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

py::dict query_sliced_py(std::string matrix_folder, std::string db_folder, std::string row_file, std::string col_file) {
    std::vector<string> identifiers;
    unordered_map<string, int> id_to_index = pc_mat::load_vector_identifiers(db_folder, identifiers);
    
    std::vector<int32_t> row_query_vec, col_query_vec;
    std::vector<std::string> row_vec, col_vec;
    
    row_query_vec = pc_mat::read_queries_from_file(row_file, id_to_index, row_vec);
    col_query_vec = pc_mat::read_queries_from_file(col_file, id_to_index, col_vec);

    int total_vectors = identifiers.size();

    std::vector<float> vector_norms;
    pc_mat::load_vector_norms(db_folder, vector_norms);

    std::vector<std::vector<float>> results = pc_mat::query_sliced(matrix_folder, row_query_vec, col_query_vec, total_vectors, vector_norms);
    
    py::list row_list, col_list;
    for(const auto& row: row_vec) row_list.append(row);
    for(const auto& col: col_vec) col_list.append(col);
    
    py::dict jaccard_dict;
    for(size_t i=0; i<results.size(); i++){
        py::list jaccard_list;
        std::string row = row_vec[i];
        for(size_t j=0; j<results[i].size(); j++){
            jaccard_list.append(results[i][j]);
        }
        jaccard_dict[row.c_str()] = jaccard_list;
    }
    
    py::dict final_result;
    final_result["row-list"] = row_list;
    final_result["col-list"] = col_list;
    final_result["jac-dict"] = jaccard_dict;
    return final_result;
}

PYBIND11_MODULE(read_pc_mat_module, m) {
    m.doc() = "Module for querying pairwise comparison matrix";
    
    m.def("query", 
        &query_py, 
        py::arg("matrix_folder"), py::arg("db_folder"), py::arg("query_file"),
        "Compute neighbors for queries in the given matrix folder, database folder and query file / ids;"
        " returns a list of dictionaries with neighbor IDs and jaccard similarities."
    );

    m.def("query_sliced",
        &query_sliced_py, 
        py::arg("matrix_folder"), py::arg("db_folder"), py::arg("row_file"), py::arg("col_file"),
        "Compute neighbors for queries in the given matrix folder, database folder and from the corresponding row-col files;"
        " returns a dictionary containing row, col IDS and their corresponding jaccard similarities."
    );
}