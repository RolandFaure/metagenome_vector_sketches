#include "read_pc_mat.h"
#include "clipp.h"
#include <cassert>

namespace fs = std::filesystem;

void query_nearest_neighbors(std::string matrix_folder, std::string query_file,
        std::vector<std::string>& query_ids_str, bool write_to_file, bool show_all_neighbors, int64_t top_n){
    auto start = std::chrono::high_resolution_clock::now();
    vector<pc_mat::Result> all_results = pc_mat::query(matrix_folder, query_file, query_ids_str);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Query completed in " << elapsed.count() << " seconds.\n" << std::endl;
    
    for(int i=0; i< all_results.size(); i++){
        
        const pc_mat::Result& res = all_results[i];
        std::cout << "Query: " << res.self_id << " #Neighbors: "<<res.neighbor_ids.size()<< std::endl;
        int64_t num_neighbors_to_show = show_all_neighbors ? 
                    res.neighbor_ids.size()
                    :std::min<int64_t>(top_n, res.neighbor_ids.size());
        std::cout << "Top " << num_neighbors_to_show << " neighbors:\n";
        std::ofstream out;
        if(write_to_file) {
            std::string nfn = res.self_id+".neighbors.txt";
            out.open(nfn.c_str());
            out<<"ID Jaccard\n";
        }
        for (size_t j = 0; j < num_neighbors_to_show; ++j) {
            std::cout <<j+1<< ". Neighbor: " << res.neighbor_ids[j]
                 << " Jaccard Similarity: " << res.jaccard_similarities[j] << endl;
            if(write_to_file) out<<res.neighbor_ids[j]<<" "<<res.jaccard_similarities[j]<<std::endl;
        }
        std::cout << std::endl;
        out.close();
    }
}

void query_sliced_matrix(std::string matrix_folder, std::string row_file, std::string col_file,
        bool write_to_file){
    std::vector<std::string> row_vec;
    std::vector<std::string> col_vec;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float> > all_results = pc_mat::query_sliced(matrix_folder, row_file, col_file, row_vec, col_vec);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Query completed in " << elapsed.count() << " seconds.\n" << std::endl;
    
    std::ofstream out;
    if(write_to_file) {
        std::string nfn = "sliced.neighbors.csv";
        out.open(nfn.c_str(), std::ios::out);
        out<<"Accession,";
        for(size_t i=0; i<col_vec.size(); i++){
            out<<col_vec[i]<<",";
        } 
        out<<"\n";
    }

    std::cout<<" \t";
    for(size_t i=0; i<col_vec.size(); i++){
        std::cout<<col_vec[i]<<"\t";
    } 
    std::cout<<std::endl;
    for(int i=0; i< all_results.size(); i++){
        std::vector<float> & res = all_results[i];
        std::cout<<row_vec[i]<<"\t";
        if(write_to_file) out<<row_vec[i]<<",";
        for (size_t j = 0; j < res.size(); ++j) {
            std::cout<<res[j]<<"\t";
            if(write_to_file) out<<res[j]<<",";
        }
        std::cout << std::endl;
        if(write_to_file) {
            out<<"\n";
        }
    }
    if(write_to_file) out.close();
}


int main(int argc, char* argv[]) {

    // Command line arguments
    string matrix_folder;
    string query_file;
    std::string row_file, col_file;
    // string neighbor_fn = "neighbors.txt";
    uint32_t top_n = 10;
    vector<string> query_ids_str;
    bool read_from_stdin = false;
    bool show_help = false;
    bool write_to_file = false;
    bool show_all_neighbors = false;
    
    bool use_query_file = false;
    bool use_query_ids = false;
    bool use_row_col_files = false;

    auto cli = (
        clipp::option("--matrix_folder") & clipp::value("folder", matrix_folder),
        (
            (clipp::option("--query_file").set(use_query_file) & clipp::value("file", query_file)) |
            (clipp::option("--query_ids").set(use_query_ids) & clipp::values("ids", query_ids_str)) |
            (
            clipp::option("--row_file").set(use_row_col_files) & clipp::value("row", row_file) &
            clipp::option("--col_file") & clipp::value("col", col_file)
            )
            // | clipp::option("--stdin").set(read_from_stdin)
        ),
        clipp::option("--top") & clipp::value("ids", top_n),
        clipp::option("--write_to_file").set(write_to_file),
        clipp::option("--show_all").set(show_all_neighbors),
        clipp::option("--help").set(show_help)
    );

    if (!clipp::parse(argc, argv, cli) || show_help) {
        cout << "Query Ava Matrix - Find neighbors in pairwise similarity matrix\n\n";
        cout << "Usage:\n" << clipp::usage_lines(cli, argv[0]) << "\n\n";
        cout << "Options:\n";
        cout << "  --matrix_folder  Folder containing the pairwise matrix files\n";
        cout << "  --query_file     File containing query IDs (one per line)\n";
        cout << "  --query_ids      Query IDs as command line arguments (numeric indices or identifiers)\n";
        cout << "  --row_file     File containing query row IDs (one per line)\n";
        cout << "  --col_file     File containing query col IDs (one per line)\n";
        // cout << "  --stdin          Read query IDs from standard input\n";
        cout << "  --top           Number of top jaccard values to show\n";
        cout << "  --write_to_file  Whether to write neighbor results to file (named after query or for row-col case \"sliced.neighbors.csv\")\n";
        cout << "  --show_all  Whether to show all neighbors instead of top N\n";
        cout << "  --help           Show this help message\n\n";
        cout << "Examples:\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_ids 10 25 42\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_ids SRR123456 SRR789012\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_file queries.txt\n";
        // cout << "  echo -e \"SRR123456\\n25\\nSRR789012\" | " << argv[0] << " --matrix_folder ./results --stdin\n";
        return show_help ? 0 : 1;
    }

    if(use_query_file || use_query_ids){
        query_nearest_neighbors(matrix_folder, query_file, query_ids_str, write_to_file, show_all_neighbors, top_n);
    }
    else if(use_row_col_files){
        if(row_file.empty() || col_file.empty()){
            std::cerr<<"Either row or col file is not specified. Aborting...\n";
            exit(-1);    
        }
        query_sliced_matrix(matrix_folder, row_file, col_file, write_to_file);
    }
    else{
        std::cerr<<"No query types specified. Aborting...\n";
        exit(-1);
    }
    
    return 0;
}
