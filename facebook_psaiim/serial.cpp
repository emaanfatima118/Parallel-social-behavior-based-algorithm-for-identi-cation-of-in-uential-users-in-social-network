#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <set>
#include <string>
#include <filesystem>
#include <algorithm>
#include <queue>
#include <cmath>
#include <random>
#include <ctime>
#include <regex>
#include <climits>

namespace fs = std::filesystem;
using namespace std;

// Structure to represent the graph
struct Graph {
    unordered_map<int, vector<int>> adjList;
    set<int> nodes;
    int nodeCount = 0;
    int edgeCount = 0;

    void addEdge(int u, int v) {
        adjList[u].push_back(v);
        adjList[v].push_back(u);
        nodes.insert(u);
        nodes.insert(v);
    }
void addNode(int node) {
        nodes.insert(node);
        if (adjList.find(node) == adjList.end()) {
            adjList[node] = vector<int>();
        }
    }
    
    // Modified build method that also takes explicit nodes
    void buildFromEdgesAndNodes(const set<pair<int, int>>& edges, const set<int>& extraNodes) {
        // Add all explicit nodes first
        for (int node : extraNodes) {
            addNode(node);
        }
        
        // Then add all edges
        for (const auto& edge : edges) {
            addEdge(edge.first, edge.second);
        }
        
        nodeCount = nodes.size();
        edgeCount = edges.size();
    }
    void buildFromEdges(const set<pair<int, int>>& edges) {
        for (const auto& edge : edges) {
            addEdge(edge.first, edge.second);
        }
        nodeCount = nodes.size();
        edgeCount = edges.size();
    }

    // Merge another graph into this one
    void mergeGraph(const Graph& other) {
        for (const auto& [node, neighbors] : other.adjList) {
            for (int neighbor : neighbors) {
                if (node < neighbor) { // Avoid duplicate edges
                    addEdge(node, neighbor);
                }
            }
        }
        nodeCount = nodes.size();
        // Recalculate edge count to avoid duplicates
        edgeCount = 0;
        for (const auto& [node, neighbors] : adjList) {
            edgeCount += neighbors.size();
        }
        edgeCount /= 2; // Each edge is counted twice in undirected graph
    }

    // BFS to find shortest paths
    unordered_map<int, int> bfs(int source) const {
        unordered_map<int, int> distances;
        queue<int> q;
        
        for (int node : nodes) {
            distances[node] = -1; // -1 indicates unreachable
        }
        
        distances[source] = 0;
        q.push(source);
        
        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            
            if (adjList.find(curr) != adjList.end()) {
                for (int neighbor : adjList.at(curr)) {
                    if (distances[neighbor] == -1) {
                        distances[neighbor] = distances[curr] + 1;
                        q.push(neighbor);
                    }
                }
            }
        }
        
        return distances;
    }

    // Calculate graph diameter (longest shortest path)
    pair<int, pair<int, int>> calculateDiameter() const {
        int maxDist = 0;
        int start = -1, end = -1;
        
        // Use a random sample of nodes for large graphs
        vector<int> sampleNodes;
        if (nodes.size() > 100) {
            sampleNodes.reserve(100);
            vector<int> allNodes(nodes.begin(), nodes.end());
            random_device rd;
            mt19937 gen(rd());
            shuffle(allNodes.begin(), allNodes.end(), gen);
            sampleNodes.insert(sampleNodes.end(), allNodes.begin(), allNodes.begin() + min(100, (int)allNodes.size()));
        } else {
            sampleNodes.insert(sampleNodes.end(), nodes.begin(), nodes.end());
        }
        
        for (int node : sampleNodes) {
            auto distances = bfs(node);
            for (const auto& [dest, dist] : distances) {
                if (dist > maxDist && dist != -1) {
                    maxDist = dist;
                    start = node;
                    end = dest;
                }
            }
        }
        
        return {maxDist, {start, end}};
    }

    // Calculate average clustering coefficient
    double calculateClusteringCoefficient(int sampleSize = 100) const {
        double totalCoeff = 0.0;
        int sampledNodes = 0;
        
        // Sample nodes for large graphs
        vector<int> nodesToCheck;
        if (nodes.size() > sampleSize) {
            vector<int> allNodes(nodes.begin(), nodes.end());
            random_device rd;
            mt19937 gen(rd());
            shuffle(allNodes.begin(), allNodes.end(), gen);
            nodesToCheck.insert(nodesToCheck.end(), allNodes.begin(), allNodes.begin() + sampleSize);
        } else {
            nodesToCheck.insert(nodesToCheck.end(), nodes.begin(), nodes.end());
        }
        
        for (int node : nodesToCheck) {
            if (adjList.find(node) == adjList.end() || adjList.at(node).size() < 2) {
                continue; // Skip nodes with fewer than 2 neighbors
            }
            
            const vector<int>& neighbors = adjList.at(node);
            int possibleConnections = neighbors.size() * (neighbors.size() - 1) / 2;
            int actualConnections = 0;
            
            for (size_t i = 0; i < neighbors.size(); ++i) {
                for (size_t j = i + 1; j < neighbors.size(); ++j) {
                    int u = neighbors[i];
                    int v = neighbors[j];
                    
                    // Check if edge exists between neighbors
                    if (find(adjList.at(u).begin(), adjList.at(u).end(), v) != adjList.at(u).end()) {
                        actualConnections++;
                    }
                }
            }
            
            double nodeCoeff = possibleConnections > 0 ? 
                static_cast<double>(actualConnections) / possibleConnections : 0;
            totalCoeff += nodeCoeff;
            sampledNodes++;
        }
        
        return sampledNodes > 0 ? totalCoeff / sampledNodes : 0;
    }

    // Calculate degree distribution
    unordered_map<int, int> calculateDegreeDistribution() const {
        unordered_map<int, int> distribution;
        
        for (const auto& [node, neighbors] : adjList) {
            distribution[neighbors.size()]++;
        }
        
        return distribution;
    }

    // Node centrality metrics
    unordered_map<int, double> calculateBetweennessCentrality(int sampleSize = 100) const {
        unordered_map<int, double> betweenness;
        
        // Initialize betweenness to 0 for all nodes
        for (int node : nodes) {
            betweenness[node] = 0.0;
        }
        
        // Sample nodes for large graphs
        vector<int> sourceNodes;
        if (nodes.size() > sampleSize) {
            vector<int> allNodes(nodes.begin(), nodes.end());
            random_device rd;
            mt19937 gen(rd());
            shuffle(allNodes.begin(), allNodes.end(), gen);
            sourceNodes.insert(sourceNodes.end(), allNodes.begin(), allNodes.begin() + sampleSize);
        } else {
            sourceNodes.insert(sourceNodes.end(), nodes.begin(), nodes.end());
        }
        
        for (int source : sourceNodes) {
            // BFS to find shortest paths
            unordered_map<int, vector<int>> predecessors;
            unordered_map<int, int> distance;
            unordered_map<int, int> numShortestPaths;
            queue<int> q;
            
            for (int node : nodes) {
                distance[node] = -1;
                numShortestPaths[node] = 0;
            }
            
            distance[source] = 0;
            numShortestPaths[source] = 1;
            q.push(source);
            
            while (!q.empty()) {
                int curr = q.front();
                q.pop();
                
                if (adjList.find(curr) != adjList.end()) {
                    for (int neighbor : adjList.at(curr)) {
                        // Found new shortest path to neighbor
                        if (distance[neighbor] == -1) {
                            distance[neighbor] = distance[curr] + 1;
                            q.push(neighbor);
                        }
                        
                        // If this is a shortest path to neighbor
                        if (distance[neighbor] == distance[curr] + 1) {
                            numShortestPaths[neighbor] += numShortestPaths[curr];
                            predecessors[neighbor].push_back(curr);
                        }
                    }
                }
            }
            
            // Calculate dependency (contribution to betweenness)
            unordered_map<int, double> dependency;
            
            // Process nodes in order of decreasing distance from source
            vector<int> orderedNodes;
            for (const auto& [node, dist] : distance) {
                if (dist > 0) { // Exclude source and unreachable nodes
                    orderedNodes.push_back(node);
                }
            }
            
            sort(orderedNodes.begin(), orderedNodes.end(), 
                 [&distance](int a, int b) { return distance[a] > distance[b]; });
            
            for (int node : orderedNodes) {
                for (int pred : predecessors[node]) {
                    double factor = static_cast<double>(numShortestPaths[pred]) / numShortestPaths[node];
                    dependency[pred] += factor * (1.0 + dependency[node]);
                }
                
                if (node != source) {
                    betweenness[node] += dependency[node];
                }
            }
        }
        
        // Normalize by the sample size
        double scaleFactor = static_cast<double>(nodes.size()) / sourceNodes.size();
        for (auto& [node, centrality] : betweenness) {
            centrality *= scaleFactor;
        }
        
        return betweenness;
    }

    // Find influencers based on various centrality metrics
    vector<int> findInfluencers(int k, const string& metric = "degree") const {
        vector<pair<int, double>> nodeCentrality;
        
        if (metric == "degree") {
            // Degree centrality
            for (const auto& [node, neighbors] : adjList) {
                nodeCentrality.push_back({node, static_cast<double>(neighbors.size())});
            }
        } else if (metric == "betweenness") {
            // Betweenness centrality (computationally expensive for large graphs)
            auto betweenness = calculateBetweennessCentrality(min(100, (int)nodes.size()));
            for (const auto& [node, centrality] : betweenness) {
                nodeCentrality.push_back({node, centrality});
            }
        } else if (metric == "pagerank") {
            // PageRank-like centrality
            double dampingFactor = 0.85;
            unordered_map<int, double> pageRank;
            unordered_map<int, double> newPageRank;
            
            // Initialize with uniform distribution
            double initValue = 1.0 / nodes.size();
            for (int node : nodes) {
                pageRank[node] = initValue;
            }
            
            // Iterate until convergence or max iterations
            for (int iter = 0; iter < 20; ++iter) {
                for (int node : nodes) {
                    newPageRank[node] = (1.0 - dampingFactor) / nodes.size();
                    
                    // Sum incoming PageRank contributions
                    if (adjList.find(node) != adjList.end()) {
                        for (int inNeighbor : adjList.at(node)) {
                            if (adjList.find(inNeighbor) != adjList.end()) {
                                int outDegree = adjList.at(inNeighbor).size();
                                newPageRank[node] += dampingFactor * pageRank[inNeighbor] / outDegree;
                            }
                        }
                    }
                }
                
                // Update PageRank values
                pageRank = newPageRank;
            }
            
            for (const auto& [node, rank] : pageRank) {
                nodeCentrality.push_back({node, rank});
            }
        }
        
        // Sort by centrality in descending order
        sort(nodeCentrality.begin(), nodeCentrality.end(), 
             [](const pair<int, double>& a, const pair<int, double>& b) {
                 return a.second > b.second;
             });
        
        vector<int> influencers;
        for (int i = 0; i < min(k, (int)nodeCentrality.size()); ++i) {
            influencers.push_back(nodeCentrality[i].first);
        }
        
        return influencers;
    }

    // Independent Cascade Model for influence propagation simulation
    int simulateICModel(const vector<int>& seeds, double probability, int iterations = 10) const {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.0, 1.0);
        
        int totalActivated = 0;
        
        for (int iter = 0; iter < iterations; ++iter) {
            set<int> activated(seeds.begin(), seeds.end());
            set<int> frontier(seeds.begin(), seeds.end());
            set<int> newFrontier;
            
            while (!frontier.empty()) {
                newFrontier.clear();
                
                for (int u : frontier) {
                    if (adjList.find(u) != adjList.end()) {
                        for (int v : adjList.at(u)) {
                            if (activated.find(v) == activated.end() && dis(gen) <= probability) {
                                activated.insert(v);
                                newFrontier.insert(v);
                            }
                        }
                    }
                }
                
                frontier = newFrontier;
            }
            
            totalActivated += activated.size();
        }
        
        return totalActivated / iterations;
    }
};
set<int> readNodes(const string& filepath) {
    ifstream file(filepath);
    set<int> nodes;
    
    if (!file.is_open()) {
        cerr << "Error opening nodes file: " << filepath << endl;
        return nodes;
    }
    
    int node;
    while (file >> node) {
        nodes.insert(node);
    }
    return nodes;
}
// Edge list
set<pair<int, int>> readEdges(const string& filepath) {
    ifstream file(filepath);
    set<pair<int, int>> edges;
    
    if (!file.is_open()) {
        cerr << "Error opening edges file: " << filepath << endl;
        return edges;
    }
    
    int u, v;
    while (file >> u >> v) {
        if (u > v) swap(u, v); // Keep consistent order for undirected
        edges.insert({u, v});
    }
    return edges;
}

// Get all ego node IDs from a directory
vector<string> getEgoNodeIds(const string& dataDir) {
    vector<string> egoIds;
    regex edgeFilePattern("([0-9]+)\\.edges");
    
    try {
        for (const auto& entry : fs::directory_iterator(dataDir)) {
            string filename = entry.path().filename().string();
            smatch match;
            if (regex_search(filename, match, edgeFilePattern) && match.size() > 1) {
                egoIds.push_back(match[1].str());
            }
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << endl;
    }
    
    return egoIds;
}

// Process multiple ego networks and combine them
Graph buildCombinedGraph(const string& dataDir, const vector<string>& egoIds) {
    Graph combinedGraph;
    set<pair<int, int>> globalEdges;
    set<int> globalNodes;
    
    for (const string& egoId : egoIds) {
        // Process edge file
        string edgeFile = dataDir + "/" + egoId + ".edges";
        auto edges = readEdges(edgeFile);
        cout << "Loaded " << edges.size() << " edges from ego network " << egoId << endl;
        
        for (const auto& edge : edges) {
            globalEdges.insert(edge);
        }
        
        // Process feature/node file if it exists
        string nodeFile = dataDir + "/" + egoId + ".feat";
        if (fs::exists(nodeFile)) {
            auto nodes = readNodes(nodeFile);
            cout << "Loaded " << nodes.size() << " nodes from ego network " << egoId << endl;
            globalNodes.insert(nodes.begin(), nodes.end());
        }
        
        // Also add the ego node itself
        globalNodes.insert(stoi(egoId));
    }
    
    // Add all circle nodes if available
    string circlesDir = dataDir + "/circles";
    if (fs::exists(circlesDir)) {
        for (const auto& entry : fs::directory_iterator(circlesDir)) {
            string filename = entry.path().filename().string();
            if (filename.find(".circles") != string::npos) {
                auto nodes = readNodes(entry.path().string());
                globalNodes.insert(nodes.begin(), nodes.end());
            }
        }
    }
    
    cout << "Total unique edges across all ego networks: " << globalEdges.size() << endl;
    cout << "Total unique nodes across all ego networks: " << globalNodes.size() << endl;
    
    combinedGraph.buildFromEdgesAndNodes(globalEdges, globalNodes);
    
    return combinedGraph;
}
int main(int argc, char* argv[]) {
    string dataDir = "facebook";
    
    if (argc > 1) {
        dataDir = argv[1];
    }
    
    cout << "Processing all ego networks from directory " << dataDir << endl;
    
    // Find all ego networks
    auto egoIds = getEgoNodeIds(dataDir);
    cout << "Found " << egoIds.size() << " ego networks" << endl;
    
    if (egoIds.empty()) {
        cerr << "No ego networks found. Please check the directory path." << endl;
        return 1;
    }
    
    // Build combined graph
    Graph combinedGraph = buildCombinedGraph(dataDir, egoIds);
    
    cout << "\n=== Combined Graph Analysis ===" << endl;
    cout << "Graph built with " << combinedGraph.nodeCount << " nodes and " 
         << combinedGraph.edgeCount << " edges." << endl;
    
    // Calculate degree distribution summary
    auto degreeDistribution = combinedGraph.calculateDegreeDistribution();
    int maxDegree = 0, minDegree = INT_MAX;
    double avgDegree = 0.0;
    
    for (const auto& [degree, count] : degreeDistribution) {
        maxDegree = max(maxDegree, degree);
        minDegree = min(minDegree, degree);
        avgDegree += static_cast<double>(degree * count);
    }
    avgDegree /= combinedGraph.nodeCount;
    
    cout << "Degree statistics:" << endl;
    cout << "  Min degree: " << minDegree << endl;
    cout << "  Max degree: " << maxDegree << endl;
    cout << "  Average degree: " << avgDegree << endl;
    
    // Find top influencers using different metrics
    cout << "\n=== Influence Maximization ===" << endl;
    
    int k = 10; // Number of top influencers to find
    
    // Using degree centrality (fastest)
    cout << "Finding top " << k << " influencers based on degree centrality..." << endl;
    auto degreeInfluencers = combinedGraph.findInfluencers(k, "degree");
    
    cout << "Top influencers by degree:" << endl;
    for (int i = 0; i < (int)degreeInfluencers.size(); ++i) {
        int node = degreeInfluencers[i];
        cout << (i+1) << ". Node " << node << " with degree " << combinedGraph.adjList[node].size() << endl;
    }
    
    // Simulate influence spread with IC model
    double probability = 0.1; // Activation probability
    int spread = combinedGraph.simulateICModel(degreeInfluencers, probability, 100);
    cout << "Expected influence spread with probability " << probability 
         << ": " << spread << " nodes (" 
         << (100.0 * spread / combinedGraph.nodeCount) << "% of the network)" << endl;
    
    // Using PageRank-like centrality (better for information flow)
    cout << "\nFinding top " << k << " influencers based on PageRank..." << endl;
    auto pagerankInfluencers = combinedGraph.findInfluencers(k, "pagerank");
    
    cout << "Top influencers by PageRank:" << endl;
    for (int i = 0; i < (int)pagerankInfluencers.size(); ++i) {
        int node = pagerankInfluencers[i];
        cout << (i+1) << ". Node " << node << " with degree " << combinedGraph.adjList[node].size() << endl;
    }
    
    spread = combinedGraph.simulateICModel(pagerankInfluencers, probability, 100);
    cout << "Expected influence spread with probability " << probability 
         << ": " << spread << " nodes (" 
         << (100.0 * spread / combinedGraph.nodeCount) << "% of the network)" << endl;
    
    // Calculate betweenness centrality (if graph is not too large)
    if (combinedGraph.nodeCount <= 10000) {
        cout << "\nFinding top " << k << " influencers based on betweenness centrality..." << endl;
        cout << "(This may take a while for large graphs)" << endl;
        auto betweennessInfluencers = combinedGraph.findInfluencers(k, "betweenness");
        
        cout << "Top influencers by betweenness centrality:" << endl;
        for (int i = 0; i < (int)betweennessInfluencers.size(); ++i) {
            int node = betweennessInfluencers[i];
            cout << (i+1) << ". Node " << node << " with degree " << combinedGraph.adjList[node].size() << endl;
        }
        
        spread = combinedGraph.simulateICModel(betweennessInfluencers, probability, 100);
        cout << "Expected influence spread with probability " << probability 
             << ": " << spread << " nodes (" 
             << (100.0 * spread / combinedGraph.nodeCount) << "% of the network)" << endl;
    }
    
    // Compare overlap between different centrality metrics
    set<int> degreeSet(degreeInfluencers.begin(), degreeInfluencers.end());
    set<int> pagerankSet(pagerankInfluencers.begin(), pagerankInfluencers.end());
    
    int overlapCount = 0;
    for (int node : pagerankSet) {
        if (degreeSet.find(node) != degreeSet.end()) {
            overlapCount++;
        }
    }
    
    cout << "\nOverlap between degree and PageRank influencers: " << overlapCount 
         << " nodes (" << (100.0 * overlapCount / k) << "%)" << endl;
    
    return 0;
}
