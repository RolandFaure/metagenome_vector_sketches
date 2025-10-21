#include "read_pc_mat.h"
#include "clipp.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {

    // Command line arguments
    string matrix_folder;
    string query_file;
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
        cout << "  --help           Show this help message\n\n";
        cout << "Examples:\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_ids 10 25 42\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_ids SRR123456 SRR789012\n";
        cout << "  " << argv[0] << " --matrix_folder ./results --query_file queries.txt\n";
        cout << "  echo -e \"SRR123456\\n25\\nSRR789012\" | " << argv[0] << " --matrix_folder ./results --stdin\n";
        return show_help ? 0 : 1;
    }

    vector<pc_mat::Result> all_results = pc_mat::query(matrix_folder, query_file);

    for(int i=0; i< all_results.size(); i++){
        const pc_mat::Result& res = all_results[i];
        cout << "Query: " << res.self_id << endl;
        for (size_t j = 0; j < res.neighbor_ids.size(); ++j) {
            cout << "  Neighbor: " << res.neighbor_ids[j]
                 << " Jaccard Similarity: " << res.jaccard_similarities[j] << endl;
        }
        cout << endl;
    }
    return 0;
}
