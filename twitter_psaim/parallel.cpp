#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include <random>
#include <queue>
#include <algorithm>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <zlib.h>
#include <memory>

// Custom deleter for gzFile pointers
struct GzFileDeleter {
    void operator()(gzFile_s* file) {
        gzclose(file);
    }
};

// Structure for graph partitioning
struct GraphPartition {
    std::unordered_map<int, std::vector<int>> outgoing_edges;
    std::unordered_set<int> local_nodes;
    std::unordered_set<int> boundary_nodes;
};

// Class for the PSAIM algorithm
class PSAIM {
private:
    // Global graph (only fully populated on rank 0)
    std::unordered_map<int, std::vector<int>> global_graph;
    
    // Local partition of the graph for this MPI process
    GraphPartition local_partition;
    
    // Parameters
    int num_nodes;
    int num_edges;
    float propagation_prob;
    int seed_set_size;
    int mc_runs;
    
    // PSAIM-specific parameters
    float alpha; // Weight for social action score
    float beta;  // Weight for interest similarity score
    float gamma; // Weight for structural influence score
    
    // Node metrics for PSAIM
    std::unordered_map<int, float> social_action_scores;
    std::unordered_map<int, float> interest_scores;
    std::unordered_map<int, float> structural_scores;
    
    // MPI variables
    int rank;
    int size;
    
    // Random generator
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

public:
    PSAIM(float prop_prob = 0.1, int seed_size = 5, int monte_carlo = 1000,
          float a = 0.4, float b = 0.3, float g = 0.3) 
        : propagation_prob(prop_prob), seed_set_size(seed_size), mc_runs(monte_carlo),
          alpha(a), beta(b), gamma(g), num_nodes(0), num_edges(0) {
        
        // Initialize MPI rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Seed random generator with different seeds per process
        std::random_device rd;
        gen = std::mt19937(rd() + rank); // Different seed per process
        dis = std::uniform_real_distribution<>(0.0, 1.0);
    }
    
    // Helper function to read gzipped files
    std::string read_line_from_gz(gzFile file) {
        char buffer[1024];
        std::string result;
        
        char* res = gzgets(file, buffer, sizeof(buffer));
        if (res != Z_NULL) {
            result = std::string(buffer);
            // Remove trailing newline if present
            if (!result.empty() && result.back() == '\n') {
                result.pop_back();
            }
        }
        
        return result;
    }
    
    // Load the graph from gzipped file (only on rank 0)
    void load_graph(const std::string& filename) {
        if (rank == 0) {
            std::cout << "Process " << rank << ": Loading graph from " << filename << std::endl;
            
            // Open gzipped file
            gzFile file = gzopen(filename.c_str(), "r");
            if (file == NULL) {
                std::cerr << "Failed to open gzipped file: " << filename << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // Use smart pointer for automatic cleanup
            std::unique_ptr<gzFile_s, GzFileDeleter> file_ptr(file);
            
            std::string line;
            std::unordered_set<int> unique_nodes;
            std::unordered_map<int, int> node_degree;
            
            // Read the gzipped file line by line
            while (!(line = read_line_from_gz(file)).empty()) {
                if (line[0] == '#') continue; // Skip comments
                
                std::istringstream iss(line);
                int src, dst;
                
                if (iss >> src >> dst) {
                    global_graph[src].push_back(dst);
                    unique_nodes.insert(src);
                    unique_nodes.insert(dst);
                    
                    // Track degree for structural score calculation
                    node_degree[src]++;
                    
                    num_edges++;
                    
                    // Print progress for large files
                    if (num_edges % 1000000 == 0) {
                        std::cout << "Processed " << num_edges << " edges..." << std::endl;
                    }
                }
            }
            
            num_nodes = unique_nodes.size();
            std::cout << "Graph loaded: " << num_nodes << " nodes, " << num_edges << " edges" << std::endl;
            
            // Initialize PSAIM scores on rank 0
            initialize_psaim_scores(unique_nodes, node_degree);
        }
        
        // Broadcast the number of nodes and edges to all processes
        MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    // Initialize the social action, interest, and structural scores for PSAIM
    void initialize_psaim_scores(const std::unordered_set<int>& unique_nodes, 
                               const std::unordered_map<int, int>& node_degree) {
        if (rank == 0) {
            std::cout << "Initializing PSAIM scores..." << std::endl;
            
            // Get maximum degree for normalization
            int max_degree = 0;
            for (const auto& pair : node_degree) {
                max_degree = std::max(max_degree, pair.second);
            }
            
            // Calculate social action scores (based on activity/influence potential)
            for (int node : unique_nodes) {
                // Social action score based on outgoing edges
                float action_factor = global_graph.count(node) ? global_graph[node].size() : 0;
                social_action_scores[node] = action_factor / (float)(max_degree > 0 ? max_degree : 1);
                
                // In Twitter context, we can define interest scores based on characteristics of the network
                // For this implementation, we'll use a combination of degree and network position
                float degree_factor = node_degree.count(node) ? node_degree.at(node) : 0;
                float normalized_degree = degree_factor / (float)(max_degree > 0 ? max_degree : 1);
                
                // Generate interest score as a function of degree and some randomness to simulate topic interests
                // In a real implementation, this would use actual tweet content/topics
                interest_scores[node] = 0.3f + 0.5f * normalized_degree + 0.2f * dis(gen);
                
                // Structural scores based on normalized degree centrality
                // For Twitter, this represents how well-connected a user is
                float degree = node_degree.count(node) ? node_degree.at(node) : 0;
                structural_scores[node] = degree / (float)(max_degree > 0 ? max_degree : 1);
            }
            
            std::cout << "PSAIM scores initialized for " << unique_nodes.size() << " nodes" << std::endl;
        }
    }
    
    // Improved graph partitioning for Twitter dataset
    void partition_graph() {
        if (rank == 0) {
            std::cout << "Partitioning Twitter graph using optimized partitioning..." << std::endl;
            
            // Create partitions
            std::vector<GraphPartition> partitions(size);
            
            // First, identify high-degree nodes for more balanced partitioning
            std::vector<std::pair<int, int>> node_degrees;
            for (const auto& node_edges : global_graph) {
                int node = node_edges.first;
                int degree = node_edges.second.size();
                node_degrees.push_back({node, degree});
            }
            
            // Sort nodes by degree (descending)
            std::sort(node_degrees.begin(), node_degrees.end(), 
                    [](const auto& a, const auto& b) { return a.second > b.second; });
            
            // Distribute high-degree nodes across partitions in round-robin fashion
            std::vector<int> partition_edge_counts(size, 0);
            
            // First, assign high-degree nodes to balance load
            for (const auto& node_degree : node_degrees) {
                int node = node_degree.first;
                int degree = node_degree.second;
                
                // Find partition with minimum edge count
                int target_rank = std::min_element(partition_edge_counts.begin(), 
                                                 partition_edge_counts.end()) - 
                                 partition_edge_counts.begin();
                
                // Add node to target partition
                partitions[target_rank].local_nodes.insert(node);
                partitions[target_rank].outgoing_edges[node] = global_graph[node];
                partition_edge_counts[target_rank] += degree;
                
                // Mark nodes in other partitions that this node connects to as boundary nodes
                for (int neighbor : global_graph[node]) {
                    for (int i = 0; i < size; i++) {
                        if (i != target_rank && 
                            partitions[i].local_nodes.find(neighbor) != partitions[i].local_nodes.end()) {
                            partitions[target_rank].boundary_nodes.insert(neighbor);
                        }
                    }
                }
            }
            
            // Calculate partition stats
            std::cout << "Partition statistics:" << std::endl;
            for (int i = 0; i < size; i++) {
                std::cout << "  Rank " << i << ": " << partitions[i].local_nodes.size() 
                          << " nodes, " << partition_edge_counts[i] << " edges" << std::endl;
            }
            
            // Keep own partition
            local_partition = partitions[0];
            
            // Send partitions to other processes
            for (int i = 1; i < size; i++) {
                // Serialize partition data
                std::vector<int> partition_data;
                std::vector<float> score_data;
                
                // Add local nodes
                partition_data.push_back(partitions[i].local_nodes.size());
                for (int node : partitions[i].local_nodes) {
                    partition_data.push_back(node);
                    
                    // Add PSAIM scores for this node
                    score_data.push_back(social_action_scores[node]);
                    score_data.push_back(interest_scores[node]);
                    score_data.push_back(structural_scores[node]);
                }
                
                // Add boundary nodes
                partition_data.push_back(partitions[i].boundary_nodes.size());
                for (int node : partitions[i].boundary_nodes) {
                    partition_data.push_back(node);
                    
                    // Add PSAIM scores for boundary nodes as well
                    score_data.push_back(social_action_scores[node]);
                    score_data.push_back(interest_scores[node]);
                    score_data.push_back(structural_scores[node]);
                }
                
                // Add edges
                partition_data.push_back(partitions[i].outgoing_edges.size());
                for (const auto& node_edges : partitions[i].outgoing_edges) {
                    partition_data.push_back(node_edges.first);
                    partition_data.push_back(node_edges.second.size());
                    for (int neighbor : node_edges.second) {
                        partition_data.push_back(neighbor);
                    }
                }
                
                // Send partition size first
                int partition_size = partition_data.size();
                MPI_Send(&partition_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                
                // Send partition data
                MPI_Send(partition_data.data(), partition_size, MPI_INT, i, 1, MPI_COMM_WORLD);
                
                // Send score data
                int score_size = score_data.size();
                MPI_Send(&score_size, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                MPI_Send(score_data.data(), score_size, MPI_FLOAT, i, 3, MPI_COMM_WORLD);
            }
        } else {
            // Receive partition from rank 0
            int partition_size;
            MPI_Recv(&partition_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            std::vector<int> partition_data(partition_size);
            MPI_Recv(partition_data.data(), partition_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Receive score data
            int score_size;
            MPI_Recv(&score_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            std::vector<float> score_data(score_size);
            MPI_Recv(score_data.data(), score_size, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Deserialize partition data
            int index = 0;
            int score_index = 0;
            
            // Extract local nodes and their scores
            int local_nodes_count = partition_data[index++];
            for (int i = 0; i < local_nodes_count; i++) {
                int node = partition_data[index++];
                local_partition.local_nodes.insert(node);
                
                // Extract PSAIM scores
                social_action_scores[node] = score_data[score_index++];
                interest_scores[node] = score_data[score_index++];
                structural_scores[node] = score_data[score_index++];
            }
            
            // Extract boundary nodes and their scores
            int boundary_nodes_count = partition_data[index++];
            for (int i = 0; i < boundary_nodes_count; i++) {
                int node = partition_data[index++];
                local_partition.boundary_nodes.insert(node);
                
                // Extract PSAIM scores
                social_action_scores[node] = score_data[score_index++];
                interest_scores[node] = score_data[score_index++];
                structural_scores[node] = score_data[score_index++];
            }
            
            // Extract edges
            int edges_count = partition_data[index++];
            for (int i = 0; i < edges_count; i++) {
                int node = partition_data[index++];
                int neighbors_count = partition_data[index++];
                
                for (int j = 0; j < neighbors_count; j++) {
                    local_partition.outgoing_edges[node].push_back(partition_data[index++]);
                }
            }
        }
        
        std::cout << "Process " << rank << " received partition with " 
                  << local_partition.local_nodes.size() << " local nodes, "
                  << local_partition.boundary_nodes.size() << " boundary nodes, and "
                  << local_partition.outgoing_edges.size() << " nodes with outgoing edges." << std::endl;
    }

    // Run Independent Cascade model simulation optimized for Twitter network
    int simulate_ic(const std::unordered_set<int>& seed_set) {
        std::unordered_set<int> activated;
        std::queue<int> frontier;
        
        // Initialize with seed set
        for (int seed : seed_set) {
            activated.insert(seed);
            frontier.push(seed);
        }
        
        // Process propagation - for Twitter we might want to limit the cascade depth
        // since Twitter has high-degree nodes that could lead to very large cascades
        int max_steps = 10; // Limit cascade depth to model Twitter's attention span
        int current_step = 0;
        
        while (!frontier.empty() && current_step < max_steps) {
            // Process an entire level of the propagation at once
            int level_size = frontier.size();
            current_step++;
            
            for (int i = 0; i < level_size; i++) {
                int current = frontier.front();
                frontier.pop();
                
                // Skip if node has no outgoing edges in this partition
                if (local_partition.outgoing_edges.find(current) == local_partition.outgoing_edges.end()) {
                    continue;
                }
                
                // Try to activate neighbors with decay based on distance from source
                float step_decay = 1.0f - (0.1f * current_step); // Decay factor based on steps
                float effective_prob = propagation_prob * step_decay;
                
                for (int neighbor : local_partition.outgoing_edges[current]) {
                    // Skip already activated nodes
                    if (activated.find(neighbor) != activated.end()) {
                        continue;
                    }
                    
                    // Scale probability by structural score to model Twitter's attention economy
                    float node_influence = 1.0f;
                    if (structural_scores.find(current) != structural_scores.end()) {
                        node_influence += structural_scores[current];
                    }
                    
                    // Attempt activation with modified propagation probability
                    if (dis(gen) < effective_prob * node_influence) {
                        activated.insert(neighbor);
                        frontier.push(neighbor);
                    }
                }
            }
        }
        
        return activated.size();
    }
    
    // Evaluate influence spread of a seed set using Monte Carlo simulations
    float evaluate_influence(const std::unordered_set<int>& seed_set) {
        int total_spread = 0;
        
        // OpenMP parallel region for Monte Carlo simulations
        #pragma omp parallel reduction(+:total_spread)
        {
            #pragma omp for
            for (int i = 0; i < mc_runs; i++) {
                total_spread += simulate_ic(seed_set);
            }
        }
        
        return static_cast<float>(total_spread) / mc_runs;
    }
    
    // Calculate PSAIM combined score for a node with Twitter-specific adjustments
    float calculate_psaim_score(int node) {
        // Ensure the node has scores
        if (social_action_scores.find(node) == social_action_scores.end()) {
            return 0.0f;
        }
        
        // For Twitter, we prioritize structural influence more than in general networks
        // as well-connected users tend to have more impact
        return alpha * social_action_scores[node] + 
               beta * interest_scores[node] + 
               gamma * structural_scores[node];
    }
    
    // Calculate similarity between two nodes based on their interest profiles
    float calculate_interest_similarity(int node1, int node2) {
        // In a real Twitter implementation, this would use topic modeling from tweets
        // Here we use a simple approximation based on the synthetic interest scores
        float diff = std::abs(interest_scores[node1] - interest_scores[node2]);
        return 1.0f - diff; // Higher value means more similar interests
    }
    
    // Find k most influential nodes using the PSAIM algorithm
    std::vector<int> find_influential_nodes() {
        std::vector<int> result_seeds;
        std::unordered_set<int> current_seeds;
        
        // For Twitter, we use a batch evaluation approach to reduce communication overhead
        int batch_size = std::min(50, (int)local_partition.local_nodes.size());
        
        for (int i = 0; i < seed_set_size; i++) {
            struct BestCandidate {
                int node;
                float score;
            } local_best = {-1, -1.0f};
            
            // Local candidates - consider only local nodes for efficiency
            std::vector<int> candidates(local_partition.local_nodes.begin(), local_partition.local_nodes.end());
            
            // Don't reconsider already selected seeds
            candidates.erase(
                std::remove_if(candidates.begin(), candidates.end(),
                    [&current_seeds](int node) {
                        return current_seeds.find(node) != current_seeds.end();
                    }),
                candidates.end()
            );
            
            // If too many candidates, take a sample of high-potential ones
            if (candidates.size() > batch_size) {
                // Sort candidates by their basic PSAIM score
                std::sort(candidates.begin(), candidates.end(),
                    [this](int a, int b) {
                        return calculate_psaim_score(a) > calculate_psaim_score(b);
                    });
                
                // Keep only the top batch_size candidates
                candidates.resize(batch_size);
            }
            
            if (rank == 0) {
                std::cout << "Evaluating batch of " << candidates.size() << " candidates for seed " 
                          << (i+1) << "..." << std::endl;
            }
            
            // Evaluate each candidate in parallel
            #pragma omp parallel for
            for (size_t j = 0; j < candidates.size(); j++) {
                int candidate = candidates[j];
                
                // Get base PSAIM score for this candidate
                float psaim_base_score = calculate_psaim_score(candidate);
                
                // Adjust score based on similarity with existing seeds (diversity)
                float similarity_penalty = 0.0f;
                for (int seed : current_seeds) {
                    similarity_penalty += calculate_interest_similarity(candidate, seed);
                }
                
                if (!current_seeds.empty()) {
                    similarity_penalty /= current_seeds.size();
                }
                
                // Diversity-aware score (penalize similar candidates)
                float diversity_factor = current_seeds.empty() ? 1.0f : (1.0f - 0.3f * similarity_penalty);
                float psaim_score = psaim_base_score * diversity_factor;
                
                // Run influence simulations for promising candidates
                float influence = 0.0f;
                if (psaim_score > 0.3f) { // Only simulate for promising candidates
                    std::unordered_set<int> new_seeds = current_seeds;
                    new_seeds.insert(candidate);
                    influence = evaluate_influence(new_seeds);
                }
                
                // Combined score with Twitter-specific weights
                float combined_score = 0.6f * psaim_score + 0.4f * influence;
                
                // Update best node (with critical section to avoid race conditions)
                #pragma omp critical
                {
                    if (combined_score > local_best.score) {
                        local_best.score = combined_score;
                        local_best.node = candidate;
                    }
                }
            }
            
            // Gather best nodes from all processes
            struct {
                float score;
                int node;
                int rank;
            } local_info, global_best;
            
            local_info.score = local_best.score;
            local_info.node = local_best.node;
            local_info.rank = rank;
            
            // Use MPI_Allreduce with MPI_MAXLOC to find the best node
            MPI_Allreduce(&local_info, &global_best, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
            
            // Broadcast the best node to all processes
            int global_best_node = global_best.node;
            MPI_Bcast(&global_best_node, 1, MPI_INT, global_best.rank, MPI_COMM_WORLD);
            
            // Add the best node to results
            result_seeds.push_back(global_best_node);
            current_seeds.insert(global_best_node);
            
            // Print progress from rank 0
            if (rank == 0) {
                std::cout << "Selected seed " << (i+1) << ": Node " << global_best_node 
                          << " with score " << global_best.score 
                          << " (from rank " << global_best.rank << ")" << std::endl;
                
                // Print component scores if available
                if (social_action_scores.find(global_best_node) != social_action_scores.end()) {
                    std::cout << "  Social action: " << social_action_scores[global_best_node]
                              << ", Interest: " << interest_scores[global_best_node]
                              << ", Structural: " << structural_scores[global_best_node] << std::endl;
                }
            }
        }
        
        return result_seeds;
    }
    
    // Run the full PSAIM algorithm on Twitter data
    void run(const std::string& filename) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Step 1: Load graph (only on rank 0)
        load_graph(filename);
        
        // Step 2: Partition graph and distribute
        partition_graph();
        
        // Synchronize all processes before starting the algorithm
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "Twitter graph partitioned and distributed. Starting PSAIM algorithm..." << std::endl;
            std::cout << "Using PSAIM parameters: alpha=" << alpha << ", beta=" << beta << ", gamma=" << gamma << std::endl;
        }
        
        // Step 3: Find influential nodes using PSAIM
        std::vector<int> influential_nodes = find_influential_nodes();
        
        // Step 4: Validate final seed set with a full simulation
        if (rank == 0) {
            std::unordered_set<int> final_seed_set(influential_nodes.begin(), influential_nodes.end());
            float final_influence = evaluate_influence(final_seed_set);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            
            std::cout << "\nPSAIM for Twitter completed in " << elapsed.count() << " seconds" << std::endl;
            std::cout << "Most influential Twitter nodes:" << std::endl;
            for (int node : influential_nodes) {
                std::cout << node << " (SA=" << social_action_scores[node] 
                          << ", I=" << interest_scores[node]
                          << ", S=" << structural_scores[node] 
                          << ", Combined=" << calculate_psaim_score(node) << ")" << std::endl;
            }
            std::cout << "Expected influence spread: " << final_influence << " Twitter users" << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    // Get process rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set number of OpenMP threads
    int num_threads = 4; // Default value
    omp_set_num_threads(num_threads);
    
    if (rank == 0) {
        std::cout << "=== Parallel Social Action Influence Maximization (PSAIM) for Twitter ===" << std::endl;
        std::cout << "Running with " << size << " MPI processes and " << num_threads << " OpenMP threads per process" << std::endl;
    }
    
    // Parse command line arguments
    std::string graph_file = "twitter_combined.txt"; // Default to the Twitter dataset
    float propagation_prob = 0.1;        // Default propagation probability
    int seed_set_size = 10;              // Default seed set size (more for Twitter)
    int mc_runs = 1000;                  // Default Monte Carlo runs
    float alpha = 0.3;                   // Default alpha weight (social action) - modified for Twitter
    float beta = 0.3;                    // Default beta weight (interest)
    float gamma = 0.4;                   // Default gamma weight (structural) - higher for Twitter
    
    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--graph" || arg == "-g") {
            if (i + 1 < argc) graph_file = argv[++i];
        } else if (arg == "--prob" || arg == "-p") {
            if (i + 1 < argc) propagation_prob = std::stof(argv[++i]);
        } else if (arg == "--seeds" || arg == "-s") {
            if (i + 1 < argc) seed_set_size = std::stoi(argv[++i]);
        } else if (arg == "--mc" || arg == "-m") {
            if (i + 1 < argc) mc_runs = std::stoi(argv[++i]);
        } else if (arg == "--threads" || arg == "-t") {
            if (i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
                omp_set_num_threads(num_threads);
            }
        } else if (arg == "--alpha" || arg == "-a") {
            if (i + 1 < argc) alpha = std::stof(argv[++i]);
        } else if (arg == "--beta" || arg == "-b") {
            if (i + 1 < argc) beta = std::stof(argv[++i]);
        } else if (arg == "--gamma" || arg == "-c") {
            if (i + 1 < argc) gamma = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            if (rank == 0) {
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  -g, --graph FILE    Input gzipped graph file (default: twitter_combined.txt.gz)" << std::endl;
                std::cout << "  -p, --prob FLOAT    Propagation probability (default: 0.1)" << std::endl;
               // std::cout << "  -s, --seeds INT
                std::cout << "  -s, --seeds INT     Number of seed nodes to select (default: 5)" << std::endl;
                std::cout << "  -m, --mc INT        Number of Monte Carlo simulations (default: 1000)" << std::endl;
                std::cout << "  -t, --threads INT   Number of OpenMP threads per process (default: 4)" << std::endl;
                std::cout << "  -a, --alpha FLOAT   Weight for social action score (default: 0.4)" << std::endl;
                std::cout << "  -b, --beta FLOAT    Weight for interest similarity score (default: 0.3)" << std::endl;
                std::cout << "  -c, --gamma FLOAT   Weight for structural influence score (default: 0.3)" << std::endl;
                std::cout << "  -h, --help          Show this help message" << std::endl;
            }
            MPI_Finalize();
            return 0;
        }
    }
    
    // Validate parameters
    if (alpha + beta + gamma != 1.0f) {
        if (rank == 0) {
            std::cout << "Warning: Alpha + Beta + Gamma = " << (alpha + beta + gamma) 
                      << ", normalizing to sum to 1.0" << std::endl;
        }
        float sum = alpha + beta + gamma;
        alpha /= sum;
        beta /= sum;
        gamma /= sum;
    }
    
    // Create and run PSAIM instance
    PSAIM psaim(propagation_prob, seed_set_size, mc_runs, alpha, beta, gamma);
    psaim.run(graph_file);
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
