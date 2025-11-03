#include "read_pc_mat.h"
#include "clipp.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {

    // Command line arguments
    string matrix_folder;
    string query_file;
    string neighbor_fn = "neighbors.txt";
    uint32_t top_n = 10;
    vector<string> query_ids_str;
    bool read_from_stdin = false;
    bool show_help = false;
    
    auto cli = (
        clipp::option("--matrix_folder") & clipp::value("folder", matrix_folder),
        (
            (clipp::option("--query_file") & clipp::value("file", query_file)) |
            (clipp::option("--query_ids") & clipp::values("ids", query_ids_str)) |
            clipp::option("--stdin").set(read_from_stdin)
        ),
        clipp::option("--top") & clipp::value("ids", top_n),
        clipp::option("--write_to_file") & clipp::value("file", neighbor_fn),
        clipp::option("--help").set(show_help)
    );

    if (!clipp::parse(argc, argv, cli) || show_help) {
        cout << "Query Ava Matrix - Find neighbors in pairwise similarity matrix\n\n";
        cout << "Usage:\n" << clipp::usage_lines(cli, argv[0]) << "\n\n";
        cout << "Options:\n";
        cout << "  --matrix_folder  Folder containing the pairwise matrix files\n";
        cout << "  --query_file     File containing query IDs (one per line)\n";
        cout << "  --query_ids      Query IDs as command line arguments (numeric indices or identifiers)\n";
        cout << "  --stdin          Read query IDs from standard input\n";
        cout << "  --top           Number of top jaccard values to show\n";
        cout << "  --write_to_file  Write neighbor results to file\n";
        cout << "  --help           Show this help message\n\n";
        cout << "Examples:\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_ids 10 25 42\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_ids SRR123456 SRR789012\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_file queries.txt\n";
        cout << "  echo -e \"SRR123456\\n25\\nSRR789012\" | " << argv[0] << " --matrix_folder ./results --stdin\n";
        return show_help ? 0 : 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    vector<pc_mat::Result> all_results = pc_mat::query(matrix_folder, query_file);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Query completed in " << elapsed.count() << " seconds.\n" << std::endl;
    
    for(int i=0; i< all_results.size(); i++){
        std::string nfn = "query_"+std::to_string(i+1)+"_"+neighbor_fn;
        std::ofstream out(nfn.c_str());
        const pc_mat::Result& res = all_results[i];
        std::cout << "Query: " << res.self_id << " #Neighbors: "<<res.neighbor_ids.size()<< std::endl;
        int64_t num_neighbors_to_show = std::min<int64_t>(top_n, res.neighbor_ids.size());
        std::cout << "Top " << num_neighbors_to_show << " neighbors:\n";
        out<<"ID Jaccard\n";
        for (size_t j = 0; j < num_neighbors_to_show; ++j) {
            std::cout <<j+1<< ". Neighbor: " << res.neighbor_ids[j]
                 << " Jaccard Similarity: " << res.jaccard_similarities[j] << endl;
            out<<res.neighbor_ids[j]<<" "<<res.jaccard_similarities[j]<<std::endl;
        }
        std::cout << std::endl;
        out.close();
    }
    return 0;
}
