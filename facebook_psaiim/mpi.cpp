#include <mpi.h>
#include <metis.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <random>
#include <algorithm>
#include <string>
#include <utility>
#include <functional>
#include <chrono>
#include <filesystem>
#include <set>

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

struct Graph {
    std::unordered_map<int, std::vector<int>> adjacency_list;
    std::unordered_map<std::pair<int, int>, double, pair_hash> semantic_weights;
    std::unordered_map<int, std::vector<int>> features;
    std::vector<int> nodes;
};

double cosine_similarity(const std::vector<int>& a, const std::vector<int>& b) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return (norm_a > 0 && norm_b > 0) ? (dot / (std::sqrt(norm_a) * std::sqrt(norm_b))) : 0.0;
}

std::vector<std::string> load_feature_names(const std::string& directory);
bool load_facebook_data(const std::string& directory, Graph& graph);
int influence_spread(const Graph& graph, int seed, double beta, std::mt19937& gen, 
                   std::unordered_set<int>& influenced_nodes);
std::vector<std::pair<int, double>> calculate_influence(const Graph& graph, const std::vector<int>& candidates,
                                                  int num_simulations, double beta);
std::vector<int> serialize_graph(const Graph& graph);
Graph deserialize_graph(const std::vector<int>& serialized);
std::vector<int> partition_graph(const Graph& graph, int num_parts);
Graph create_subgraph(const Graph& full_graph, const std::vector<int>& node_partitions,
                    int partition_id, const std::vector<int>& nodes);

std::vector<std::string> load_feature_names(const std::string& directory) {
    std::vector<std::string> feature_names;

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        std::string filename = entry.path().filename().string();
        if (filename.find(".featnames") != std::string::npos) {
            std::ifstream file(entry.path());
            if (file) {
                std::string line;
                while (std::getline(file, line)) {
                    std::istringstream iss(line);
                    int feat_id;
                    std::string name;
                    if (iss >> feat_id >> name) {
                        feature_names.push_back(name);
                    }
                }
                break; 
            }
        }
    }

    return feature_names;
}

bool load_facebook_data(const std::string& directory, Graph& graph) {
    std::set<int> all_nodes;

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        std::string filename = entry.path().filename().string();
        
        if (filename.find(".edges") != std::string::npos) {
            std::ifstream edge_stream(entry.path());
            if (!edge_stream) {
                std::cerr << "Warning: Failed to open edge file: " << filename << std::endl;
                continue;
            }
            
            std::cout << "Loading edges from: " << filename << std::endl;
            
            int u, v;
            while (edge_stream >> u >> v) {
                graph.adjacency_list[u].push_back(v);
                graph.adjacency_list[v].push_back(u); // Undirected graph
                
                all_nodes.insert(u);
                all_nodes.insert(v);
            }
        }
    }

    graph.nodes.assign(all_nodes.begin(), all_nodes.end());
    std::sort(graph.nodes.begin(), graph.nodes.end());

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        std::string filename = entry.path().filename().string();
        
        if (filename.find(".feat") != std::string::npos && filename.find(".egofeat") == std::string::npos) {
            std::ifstream feat_stream(entry.path());
            if (!feat_stream) {
                std::cerr << "Warning: Failed to open feature file: " << filename << std::endl;
                continue;
            }
            
            std::cout << "Loading features from: " << filename << std::endl;
            
            std::string line;
            while (std::getline(feat_stream, line)) {
                std::istringstream iss(line);
                int node_id;
                iss >> node_id;
                
                std::vector<int> feature_vector;
                int feat_val;
                while (iss >> feat_val) {
                    feature_vector.push_back(feat_val);
                }
                
                if (!feature_vector.empty()) {
                    graph.features[node_id] = feature_vector;
                }
            }
        }
        if (filename.find(".egofeat") != std::string::npos) {
            std::ifstream egofeat_stream(entry.path());
            if (!egofeat_stream) {
                std::cerr << "Warning: Failed to open egofeat file: " << filename << std::endl;
                continue;
            }
    
            std::string basename = entry.path().filename().string();
            int ego_id = std::stoi(basename.substr(0, basename.find(".")));
    
            std::string line;
            while (std::getline(egofeat_stream, line)) {
                std::istringstream iss(line);
                std::vector<int> features;
                int feat_val;
                while (iss >> feat_val) {
                    features.push_back(feat_val);
                }
    
                if (!features.empty()) {
                    graph.features[ego_id] = features;
                }
            }
        }
        
    }
    
    std::cout << "Computing semantic weights between nodes..." << std::endl;
    for (const auto& edge_pair : graph.adjacency_list) {
        int u = edge_pair.first;
        for (int v : edge_pair.second) {
            if (u < v) { 
                if (graph.features.count(u) && graph.features.count(v)) {
                    double sim = cosine_similarity(graph.features[u], graph.features[v]);
                    std::pair<int, int> edge1 = {u, v};
                    std::pair<int, int> edge2 = {v, u};
                    graph.semantic_weights[edge1] = sim;
                    graph.semantic_weights[edge2] = sim; 
                } else {
                   
                    std::pair<int, int> edge1 = {u, v};
                    std::pair<int, int> edge2 = {v, u};
                    graph.semantic_weights[edge1] = 0.5;
                    graph.semantic_weights[edge2] = 0.5;
                }
            }
        }
    }

    return !graph.nodes.empty();
}


int influence_spread(const Graph& graph, int seed, double beta, std::mt19937& gen,
                    std::unordered_set<int>& influenced_nodes) {
    std::unordered_set<int> active = { seed };
    std::unordered_set<int> newly_active = { seed };
    influenced_nodes.insert(seed);

    while (!newly_active.empty()) {
        std::unordered_set<int> next_active;
        for (int u : newly_active) {
            if (!graph.adjacency_list.count(u)) {
                continue;
            }
            
            for (int v : graph.adjacency_list.at(u)) {
                if (active.count(v)) continue;
                
                double weight = 0.5; 
                std::pair<int, int> edge = {u, v};
                if (graph.semantic_weights.count(edge)) {
                    weight = graph.semantic_weights.at(edge);
                }
                
                double p = beta * weight;
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                if (dist(gen) < p) {
                    next_active.insert(v);
                    active.insert(v);
                    influenced_nodes.insert(v);
                }
            }
        }
        newly_active = std::move(next_active);
    }

    return static_cast<int>(active.size());
}

std::vector<std::pair<int, double>> calculate_influence(const Graph& graph, const std::vector<int>& candidates,
                                                  int num_simulations, double beta) {
    std::vector<std::pair<int, double>> influence_scores;
    std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());

    std::cout << "Calculating influence for " << candidates.size() << " candidate nodes..." << std::endl;
    int counter = 0;

    for (int seed : candidates) {
        double total_spread = 0.0;
        for (int sim = 0; sim < num_simulations; ++sim) {
            std::unordered_set<int> influenced;
            total_spread += influence_spread(graph, seed, beta, gen, influenced);
        }
        double avg_spread = total_spread / num_simulations;
        influence_scores.push_back({seed, avg_spread});
        
        // Progress report
        counter++;
        if (counter % 100 == 0 || counter == static_cast<int>(candidates.size())) {
            std::cout << "Processed " << counter << "/" << candidates.size() << " candidates..." << std::endl;
        }
    }

    std::sort(influence_scores.begin(), influence_scores.end(), 
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second > b.second; });

    return influence_scores;
}

// --- Serialize Graph for MPI Communication ---
std::vector<int> serialize_graph(const Graph& graph) {
    std::vector<int> serialized;

    serialized.push_back(graph.nodes.size());

    for (int node : graph.nodes) {
        serialized.push_back(node);
    }

    serialized.push_back(graph.adjacency_list.size());
    for (const auto& adj_pair : graph.adjacency_list) {
        serialized.push_back(adj_pair.first);
        serialized.push_back(adj_pair.second.size());
        for (int neighbor : adj_pair.second) {
            serialized.push_back(neighbor);
        }
    }

 
    serialized.push_back(graph.features.size());
    for (const auto& feat_pair : graph.features) {
        serialized.push_back(feat_pair.first);
        serialized.push_back(feat_pair.second.size());
        for (int feat_val : feat_pair.second) {
            serialized.push_back(feat_val);
        }
    }

    // Store semantic weights (as integers with scaling)
    int scale_factor = 10000; // To preserve floating point precision
    serialized.push_back(graph.semantic_weights.size());
    for (const auto& weight_pair : graph.semantic_weights) {
        serialized.push_back(weight_pair.first.first);
        serialized.push_back(weight_pair.first.second);
        serialized.push_back(static_cast<int>(weight_pair.second * scale_factor));
    }

    return serialized;
}

Graph deserialize_graph(const std::vector<int>& serialized) {
    Graph graph;
    int idx = 0;

    int num_nodes = serialized[idx++];
    for (int i = 0; i < num_nodes; ++i) {
        graph.nodes.push_back(serialized[idx++]);
    }

    int num_adj_entries = serialized[idx++];
    for (int i = 0; i < num_adj_entries; ++i) {
        int node = serialized[idx++];
        int num_neighbors = serialized[idx++];
        for (int j = 0; j < num_neighbors; ++j) {
            graph.adjacency_list[node].push_back(serialized[idx++]);
        }
    }

    int num_features = serialized[idx++];
    for (int i = 0; i < num_features; ++i) {
        int node = serialized[idx++];
        int feat_size = serialized[idx++];
        std::vector<int> features;
        for (int j = 0; j < feat_size; ++j) {
            features.push_back(serialized[idx++]);
        }
        graph.features[node] = features;
    }

    int scale_factor = 10000;
    int num_weights = serialized[idx++];
    for (int i = 0; i < num_weights; ++i) {
        int u = serialized[idx++];
        int v = serialized[idx++];
        double weight = static_cast<double>(serialized[idx++]) / scale_factor;
        std::pair<int, int> edge = {u, v};
        graph.semantic_weights[edge] = weight;
    }

    return graph;
}

std::vector<int> partition_graph(const Graph& graph, int num_parts) {
    idx_t nvtxs = graph.nodes.size(); 

    if (nvtxs == 0) {
        std::cerr << "Error: Empty graph cannot be partitioned." << std::endl;
        return std::vector<int>();
    }

    std::vector<idx_t> xadj(nvtxs + 1);
    std::vector<idx_t> adjncy;

    // Map node IDs to consecutive indices
    std::unordered_map<int, int> node_to_idx;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        node_to_idx[graph.nodes[i]] = i;
    }

    xadj[0] = 0;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        int node = graph.nodes[i];
        if (graph.adjacency_list.count(node)) {
            for (int neighbor : graph.adjacency_list.at(node)) {
                if (node_to_idx.count(neighbor)) { 
                    adjncy.push_back(node_to_idx[neighbor]);
                }
            }
        }
        xadj[i+1] = adjncy.size();
    }

    idx_t ncon = 1; 
    idx_t nparts = num_parts; 
    idx_t objval; 
    std::vector<idx_t> part(nvtxs); 

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    std::cout << "Partitioning graph with METIS: " << nvtxs << " nodes, " 
              << adjncy.size() << " edges, " << num_parts << " partitions..." << std::endl;

    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), 
                                 NULL, NULL, NULL, &nparts, NULL, NULL, 
                                 options, &objval, part.data());

    if (ret != METIS_OK) {
        std::cerr << "Error in METIS partitioning! Error code: " << ret << std::endl;
        return std::vector<int>();
    }

    std::cout << "Graph partitioned successfully. Edge-cut: " << objval << std::endl;

    std::vector<int> node_partitions(graph.nodes.size());
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        node_partitions[i] = part[i];
    }

    return node_partitions;
}

Graph create_subgraph(const Graph& full_graph, const std::vector<int>& node_partitions,
                    int partition_id, const std::vector<int>& nodes) {
    Graph subgraph;

    for (size_t i = 0; i < nodes.size(); ++i) {
        if (node_partitions[i] == partition_id) {
            int node = nodes[i];
            subgraph.nodes.push_back(node);
            
            // Add feature vector
            if (full_graph.features.count(node)) {
                subgraph.features[node] = full_graph.features.at(node);
            }
            
            // Add adjacency list (including cross-partition edges)
            if (full_graph.adjacency_list.count(node)) {
                subgraph.adjacency_list[node] = full_graph.adjacency_list.at(node);
                
                // Add semantic weights
                for (int neighbor : subgraph.adjacency_list[node]) {
                    std::pair<int, int> edge = {node, neighbor};
                    if (full_graph.semantic_weights.count(edge)) {
                        subgraph.semantic_weights[edge] = full_graph.semantic_weights.at(edge);
                    }
                }
            }
        }
    }

    return subgraph;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Parameters
    std::string data_directory = ".";  // Default to current directory
    double beta = 0.02;                // Base influence probability
    int num_simulations = 10;          // Number of Monte Carlo simulations
    int num_seeds = 10;                // Number of top seeds to report

    // Override default parameters if provided
    if (argc > 1) {
        data_directory = argv[1];
    }
    if (argc > 2) {
        beta = std::stod(argv[2]);
    }
    if (argc > 3) {
        num_simulations = std::stoi(argv[3]);
    }
    if (argc > 4) {
        num_seeds = std::stoi(argv[4]);
    }

    Graph full_graph;
    std::vector<int> node_partitions;

    if (rank == 0) {
        std::cout << "PSAIIM: Parallel Semantic-Aware Influence Maximization for Facebook SNAP Data" << std::endl;
        std::cout << "Data directory: " << data_directory << std::endl;
        std::cout << "Beta: " << beta << ", Num simulations: " << num_simulations << std::endl;
        
        if (!load_facebook_data(data_directory, full_graph)) {
            std::cerr << "Failed to load Facebook data. Exiting." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        std::cout << "Graph loaded: " << full_graph.nodes.size() << " nodes, " 
                  << full_graph.adjacency_list.size() << " adjacency entries, "
                  << full_graph.features.size() << " nodes with features" << std::endl;
        
        node_partitions = partition_graph(full_graph, num_procs);
        if (node_partitions.empty()) {
            std::cerr << "Failed to partition graph. Exiting." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        std::cout << "Graph partitioned into " << num_procs << " parts" << std::endl;
        
        for (int p = 0; p < num_procs; ++p) {
            Graph subgraph = create_subgraph(full_graph, node_partitions, p, full_graph.nodes);
            
            std::cout << "Created subgraph for partition " << p << " with " 
                      << subgraph.nodes.size() << " nodes" << std::endl;
            
            if (p == 0) {
                // Master keeps its own subgraph
                full_graph = subgraph;
            } else {
                // Send to worker processes
                std::vector<int> serialized = serialize_graph(subgraph);
                
                // Send size first
                int size = serialized.size();
                MPI_Send(&size, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                
                // Then send the data
                MPI_Send(serialized.data(), size, MPI_INT, p, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        int size;
        MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<int> serialized(size);
        MPI_Recv(serialized.data(), size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        full_graph = deserialize_graph(serialized);
        
        std::cout << "Rank " << rank << " received subgraph with " 
                  << full_graph.nodes.size() << " nodes" << std::endl;
    }

    std::vector<std::pair<int, double>> local_influence = 
        calculate_influence(full_graph, full_graph.nodes, num_simulations, beta);

    if (rank == 0) {
        std::vector<std::pair<int, double>> all_influence = local_influence;
        
        for (int p = 1; p < num_procs; ++p) {
            int size;
            MPI_Recv(&size, 1, MPI_INT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            std::vector<int> buffer(size);
            MPI_Recv(buffer.data(), size, MPI_INT, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            int scale_factor = 1000000;
            for (int i = 0; i < size; i += 2) {
                int node = buffer[i];
                double score = static_cast<double>(buffer[i+1]) / scale_factor;
                all_influence.push_back({node, score});
            }
        }
        
        std::sort(all_influence.begin(), all_influence.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << "\nTop " << num_seeds << " influential nodes:" << std::endl;
        std::cout << "Rank\tNode ID\tInfluence Score" << std::endl;
        for (int i = 0; i < std::min(num_seeds, static_cast<int>(all_influence.size())); ++i) {
            std::cout << i+1 << "\t" << all_influence[i].first << "\t" 
                      << all_influence[i].second << std::endl;
        }
        
        std::ofstream result_file("top_influential_nodes.txt");
        if (result_file) {
            result_file << "Rank\tNode ID\tInfluence Score" << std::endl;
            for (int i = 0; i < std::min(num_seeds, static_cast<int>(all_influence.size())); ++i) {
                result_file << i+1 << "\t" << all_influence[i].first << "\t" 
                          << all_influence[i].second << std::endl;
            }
            std::cout << "Results saved to top_influential_nodes.txt" << std::endl;
        }
    } else {
        std::vector<int> buffer;
        int scale_factor = 1000000;
        
        for (const auto& pair : local_influence) {
            buffer.push_back(pair.first);
            buffer.push_back(static_cast<int>(pair.second * scale_factor));
        }
        
        int size = buffer.size();
        MPI_Send(&size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(buffer.data(), size, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
