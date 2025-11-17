#include "read_pc_mat.h"
#include "clipp.h"
#include "cnpy.h"
#include <cassert>
#include <cmath>

namespace fs = std::filesystem;


void show_error_and_exit(std::string msg){
    std::cerr<<msg<<std::endl;
    std::cerr<<"Aborting...\n";
    exit(1);
}

double roundUpToTwoDecimals(double num) {
  return std::ceil(num * 100.0) / 100.0;
}

std::pair<double, std::string> get_time_unit(double total_time){
    if(total_time < 60){
        // return std::to_string(static_cast<uint64_t>( std::ceil(total_time)))+"\t seconds";
        return std::make_pair(total_time, "seconds");
        // return std::to_string(roundUpToTwoDecimals(total_time))+"\t seconds";
    }
    else if(total_time < 60*60){
        total_time/=60.0;
        // return std::to_string(static_cast<uint64_t>( std::ceil(total_time)))+"\t minutes";
        return std::make_pair(total_time, "minutes");
    }
    else{
        total_time/=(60.0*60);
        // return std::to_string(static_cast<uint64_t>( std::ceil(total_time)))+"\t hours";
        return std::make_pair(total_time, "hours");
    }
}

void query_nearest_neighbors(std::string matrix_folder, std::string db_folder, std::string query_file,
        std::vector<std::string>& query_ids_str, bool write_to_file, 
        bool show_all_neighbors, int64_t top_n, uint32_t batch_size, std::string out_fn, std::string sep, bool print_to_screen){
    
    std::vector<string> identifiers;
    unordered_map<string, int> id_to_index = pc_mat::load_vector_identifiers(db_folder, identifiers);
    
    std::vector<std::string> query_id_vec;    
    vector<int32_t> queries;
    if (!query_file.empty()) {
        queries = pc_mat::read_queries_from_file(query_file, id_to_index, query_id_vec);
    } else if (!query_ids_str.empty()) {
        // Convert command line query IDs
        for (const string& query_str : query_ids_str) {
            int index = pc_mat::parse_query_to_index(query_str, id_to_index);
            if (index >= 0) {
                queries.push_back(index);
            }
        }
    } else {
        show_error_and_exit("Error: No queries specified. Use --query_file, --query_ids");
    }

    if (queries.empty()) {
        show_error_and_exit("Error: No valid queries found");
    }
    
    
    std::vector<float> vector_norms;
    pc_mat::load_vector_norms(db_folder, vector_norms);

    int total_vectors = identifiers.size();
    std::cout<<"Total vectors loaded: " << total_vectors << endl<<endl;
    if (total_vectors <= 0) {
        show_error_and_exit("Error: Could not determine total number of vectors" );
    }
    
    // std::ofstream log_out(matrix_folder + "/neighbors_all.txt");
    std::chrono::duration<double> elapsed = std::chrono::duration<double>::zero();

    
    uint64_t start_indx = 0, end_indx;
    while(1){
        end_indx = std::min(start_indx + batch_size, queries.size());
        std::vector<int32_t> sub_queries(queries.begin()+start_indx, queries.begin()+end_indx);
        auto start = std::chrono::high_resolution_clock::now();
        vector<pc_mat::Result> all_results = pc_mat::query(matrix_folder, sub_queries, 
            vector_norms, identifiers);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed += (end - start);
        for(int i=0; i< all_results.size(); i++){
        
            const pc_mat::Result& res = all_results[i];
            // if(print_to_screen) std::cout<<start_indx + i<<" "<<res.self_id<<" "<<res.neighbor_ids.size()<<"\n";
            // log_out<<start_indx + i<<" "<<res.self_id<<" "<<res.neighbor_ids.size()<<"\n";
            if(print_to_screen) std::cout << "Query: " << res.self_id << " #Neighbors: "<<res.neighbor_ids.size()<< std::endl;
            
            std::ofstream out;
            if(write_to_file) {
                std::string nfn = res.self_id+"_"+out_fn;
                std::cout<<"Writing in file: "<<nfn<<std::endl<<std::endl;
                out.open(nfn.c_str());
                out<<"ID"+sep+"Jaccard\n";
            }

            int64_t num_neighbors_to_show = show_all_neighbors ? 
                        res.neighbor_ids.size()
                        :std::min<int64_t>(top_n, res.neighbor_ids.size());
            if(print_to_screen) std::cout << "Top " << num_neighbors_to_show << " neighbors:\n";

            for (size_t j = 0; j < num_neighbors_to_show; ++j) {
                if(print_to_screen) std::cout <<j+1<< ". Neighbor: " << res.neighbor_ids[j]
                     << " Jaccard Similarity: " << res.jaccard_similarities[j] << endl;
                if(write_to_file) out<<res.neighbor_ids[j]<<sep<<res.jaccard_similarities[j]<<std::endl;
            }
            if(print_to_screen) std::cout << std::endl;
            out.close();
        }
        auto time_unit = get_time_unit(elapsed.count());
        std::cout<<"--------- Completed\t"<<end_indx<<"\tqueries in\t"<<std::fixed << std::setprecision(2) << time_unit.first<<"\t"<<time_unit.second<<" ---------\n";
        if(end_indx == queries.size()) break;
        start_indx += batch_size;
    }    

    auto time_unit = get_time_unit(elapsed.count());

    std::cout << "Query completed in "<<std::fixed << std::setprecision(2) << time_unit.first<<"\t"<<time_unit.second<< "\n" << std::endl;
}

void query_sliced_matrix(std::string matrix_folder, std::string db_folder, std::string row_file, std::string col_file,
        bool write_to_file, std::string out_fn, uint32_t batch_size, bool print_to_screen, std::string sep){
    std::vector<string> identifiers;
    unordered_map<string, int> id_to_index = pc_mat::load_vector_identifiers(db_folder, identifiers);
    
    std::vector<int32_t> row_query_vec, col_query_vec;
    std::vector<std::string> row_vec, col_vec;
    
    row_query_vec = pc_mat::read_queries_from_file(row_file, id_to_index, row_vec);
    col_query_vec = pc_mat::read_queries_from_file(col_file, id_to_index, col_vec);

    if (row_query_vec.empty() || col_query_vec.empty()) {
        show_error_and_exit("Empty row or col accessions.");
    }
    
    std::vector<float> vector_norms;
    pc_mat::load_vector_norms(db_folder, vector_norms);

    int total_vectors = identifiers.size();
    std::cout<<"Total vectors loaded: " << total_vectors << endl<<endl;
    if (total_vectors <= 0) {
        show_error_and_exit("Error: Could not determine total number of vectors");
    }
    std::chrono::duration<double> elapsed = std::chrono::duration<double>::zero();
    uint64_t start_indx = 0, end_indx;

    std::ofstream out;
    if(write_to_file && sep != "-1") {
        std::cout<<"Writing in file: "<<out_fn<<std::endl<<std::endl;
        out.open(out_fn.c_str());
        out<<"Accession"+sep;
        for(size_t i=0; i<col_vec.size(); i++){
            out<<col_vec[i]<<sep;
        } 
        out<<"\n";
    }

    if(print_to_screen) std::cout<<"Accession\t";

    for(size_t i=0; i<col_vec.size(); i++){
        if(print_to_screen) std::cout<<col_vec[i]<<"\t";
    } 
    if(print_to_screen) std::cout<<"\n";

    
    while(1){
        end_indx = std::min(start_indx + batch_size, row_query_vec.size());
        std::vector<int32_t> row_sub_queries(row_query_vec.begin()+start_indx, row_query_vec.begin()+end_indx);
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<float> > all_results = pc_mat::query_sliced(matrix_folder, row_sub_queries, 
            col_query_vec, total_vectors, vector_norms);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed += (end - start);
        
        for(int i=0; i< all_results.size(); i++){
            std::vector<float> & res = all_results[i];
            
            if(print_to_screen) std::cout<<row_vec[start_indx + i]<<"\t";
            if(write_to_file && sep != "-1") out<<row_vec[start_indx + i]<<sep;
            
            if(print_to_screen || (write_to_file && sep != "-1")){
                for (size_t j = 0; j < res.size(); ++j) {
                    if(print_to_screen) std::cout<<res[j]<<"\t";
                    if(write_to_file && sep != "-1") out<<res[j]<<sep;
                }
            }
            
            if(write_to_file && sep == "-1"){
                if(start_indx == 0 && i == 0)
                    cnpy::npy_save(out_fn, res.data(), {1, res.size()}, "w");
                else
                    cnpy::npy_save(out_fn, res.data(), {1, res.size()}, "a");
            }
            
            if(print_to_screen) std::cout << std::endl;
            if(write_to_file && sep != "-1") out<<"\n";
        }
        
        auto time_unit = get_time_unit(elapsed.count());
        std::cout<<"--------- Completed\t"<<end_indx<<"\trows in\t"<<std::fixed << std::setprecision(2) << time_unit.first<<"\t"<<time_unit.second<<" ---------\n";
        
        if(end_indx == row_query_vec.size()) break;
        start_indx += batch_size;
    }

    auto time_unit = get_time_unit(elapsed.count());

    std::cout << "Query completed in " << std::fixed << std::setprecision(2) << time_unit.first<<"\t"<<time_unit.second<<"\n" << std::endl;

    if(write_to_file && sep != "-1") out.close();
}

std::string get_file_extension(std::string filename){
    size_t dot_pos = filename.find_last_of(".");

    if (dot_pos != std::string::npos) {
        return filename.substr(dot_pos + 1);
    }
    return "";
}


int main(int argc, char* argv[]) {

    // Command line arguments
    string matrix_folder, db_folder;
    string query_file;
    std::string row_file, col_file;
    // string neighbor_fn = "neighbors.txt";
    uint32_t top_n = 10, batch_size = 1000;
    vector<string> query_ids_str;
    bool read_from_stdin = false;
    bool show_help = false;
    bool write_to_file = false;
    std::string out_fn = "out.txt";
    bool print_to_screen = false;
    bool show_all_neighbors = false;
    
    bool use_query_file = false;
    bool use_query_ids = false;
    bool use_row_col_files = false;

    auto cli = (
        clipp::option("--matrix") & clipp::value("folder", matrix_folder),
        clipp::option("--db") & clipp::value("folder", db_folder),
        (
            (clipp::option("--query_file").set(use_query_file) & clipp::value("file", query_file)) |
            (clipp::option("--query_ids").set(use_query_ids) & clipp::values("ids", query_ids_str)) |
            (
            clipp::option("--row_file").set(use_row_col_files) & clipp::value("row", row_file) &
            clipp::option("--col_file") & clipp::value("col", col_file)
            )
            // | clipp::option("--stdin").set(read_from_stdin)
        ),
        clipp::option("--top") & clipp::value("int", top_n),
        clipp::option("--batch_size") & clipp::value("int", batch_size),
        clipp::option("--write_to_file").set(write_to_file) & clipp::value("file", out_fn),
        clipp::option("--show_all").set(show_all_neighbors),
        clipp::option("--print").set(print_to_screen),
        clipp::option("--help").set(show_help)
    );

    if (!clipp::parse(argc, argv, cli) || show_help) {
        cout << "Query Pairwise Comparison Matrix\n\n";
        cout << "Usage:\n" << clipp::usage_lines(cli, argv[0]) << "\n\n";
        cout << "Options:\n";
        cout << "  --matrix\t Folder containing the pairwise matrix files\n";
        cout << "  --db\t Folder containing the matrix meta data\n";
        cout << "  --query_file\t File containing query IDs (one per line)\n";
        cout << "  --query_ids\t Query IDs as command line arguments (numeric indices or identifiers)\n";
        cout << "  --row_file\t File containing query row IDs (one per line)\n";
        cout << "  --col_file\t File containing query col IDs (one per line)\n";
        // cout << "  --stdin          Read query IDs from standard input\n";
        cout << "  --top\t Number of top jaccard values to show [default 10]\n";
        cout << "  --batch_size\t Number of queries to process per batch [default 1000]\n";
        cout << "  --write_to_file\t Where to save the output (expected format: *.csv/*.tsv/*.npy/*npz for row-col query. *.csv/*tsv/*txt for regular query).\n";
        cout << "  --show_all\t Whether to show all neighbors instead of top N\n";
        cout << "  --print\t Whether to print the outputs to screen\n";
        cout << "  --help\t Show this help message\n\n";
        // cout << "Examples:\n";
        // cout << "  " << argv[0] << " --matrix_folder ./results --query_ids SRR123456 SRR789012\n";
        // cout << "  " << argv[0] << " --matrix_folder ./results --query_file queries.txt\n";
        return show_help ? 0 : 1;
    }

    if (matrix_folder.empty()) {
        show_error_and_exit("Error: matrix folder is required.");
    }
    if(!use_query_file && !use_query_ids && !use_row_col_files){
        show_error_and_exit("No query files given.");
    }

    if (!fs::exists(matrix_folder)) {
        show_error_and_exit("Error: Matrix folder does not exist.");
    }

    // Ensure matrix_folder ends with '/'
    if (!matrix_folder.empty() && matrix_folder.back() != '/' && matrix_folder.back() != '\\') {
        matrix_folder += '/';
    }

    if (!db_folder.empty() && db_folder.back() != '/' && db_folder.back() != '\\') {
        db_folder += '/';
    }

    if(write_to_file && out_fn.empty()){
        show_error_and_exit("No output filename given.");
    }

    if(!write_to_file) print_to_screen = true;

    if(use_query_file || use_query_ids){
        std::string file_extension = get_file_extension(out_fn);
        if(write_to_file){
            if(file_extension != "csv" && file_extension != "tsv" && file_extension != "txt"){
                show_error_and_exit("Output file extension is: "+file_extension+". Expected: csv, tsv or txt.");
            }
        }
        
        std::string sep = file_extension == "csv" ? "," : "\t";
        query_nearest_neighbors(matrix_folder, db_folder, query_file, query_ids_str, 
            write_to_file, show_all_neighbors, top_n, batch_size, out_fn, sep, print_to_screen);
    }
    else if(use_row_col_files){
        if(row_file.empty() || col_file.empty()){
            show_error_and_exit("Either row or col file is not specified.");
        }
        std::string file_extension = get_file_extension(out_fn);
        if(write_to_file){
            if(file_extension != "csv" && file_extension != "tsv" && file_extension != "npy" && file_extension != "npz"){
                show_error_and_exit("Output file extension is: "+file_extension+". Expected: csv, tsv, npy or npz.");
            }
        }
        std::string sep = "-1";
        if(file_extension == "csv" || file_extension == "tsv"){
            sep = file_extension == "csv" ? "," : "\t";
        }

        query_sliced_matrix(matrix_folder, db_folder, row_file, col_file, write_to_file, out_fn, batch_size, print_to_screen, sep);
    }
    else{
        std::cerr<<"No query types specified. Aborting...\n";
        exit(1);
    }
    
    return 0;
}
