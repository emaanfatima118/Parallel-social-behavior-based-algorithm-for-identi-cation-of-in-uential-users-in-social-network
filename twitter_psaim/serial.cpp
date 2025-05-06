#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>

using namespace std;

using Graph = unordered_map<string, vector<string>>;

// Load graph from file
Graph load_graph(const string& edge_file) {
    Graph G;
    ifstream infile(edge_file);
    string line;
    while (getline(infile, line)) {
        istringstream iss(line);
        string u, v;
        if (!(iss >> u >> v)) continue;
        G[u].push_back(v);
        if (G.find(v) == G.end()) {
            G[v] = vector<string>(); // ensure v in graph
        }
    }
    cout << "Number of nodes: " << G.size() << endl;
    return G;
}

// IC spread simulation
double simulate_ic(const Graph& G, const unordered_set<string>& seeds, double p, int mc_runs) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    double total = 0.0;
    for (int run = 0; run < mc_runs; ++run) {
        unordered_set<string> active(seeds);
        vector<string> frontier(seeds.begin(), seeds.end());

        while (!frontier.empty()) {
            vector<string> new_front;
            for (const auto& u : frontier) {
                for (const auto& v : G.at(u)) {
                    if (active.find(v) == active.end() && dis(gen) < p) {
                        active.insert(v);
                        new_front.push_back(v);
                    }
                }
            }
            frontier = move(new_front);
        }
        total += active.size();
    }
    return total / mc_runs;
}

// CELF algorithm
vector<string> celf_ic(const Graph& G, int k, double p, int mc_runs) {
    using NodeInfo = tuple<double, string, int>;
    auto cmp = [](const NodeInfo& a, const NodeInfo& b) {
        return get<0>(a) < get<0>(b); // max heap
    };
    priority_queue<NodeInfo, vector<NodeInfo>, decltype(cmp)> Q(cmp);

    for (const auto& [u, _] : G) {
        double spread = simulate_ic(G, {u}, p, mc_runs);
        Q.emplace(spread, u, 0);
    }

    unordered_set<string> S;
    vector<string> seeds;
    double spread_S = 0.0;
    int round = 1;

    while (S.size() < k && !Q.empty()) {
        auto [gain, u, last] = Q.top(); Q.pop();

        if (last < round) {
         unordered_set<string> S_with_u = S;
S_with_u.insert(u);
double new_spread = simulate_ic(G, S_with_u, p, mc_runs);

            double marginal = new_spread - spread_S;
            Q.emplace(marginal, u, round);
        } else {
            S.insert(u);
            spread_S += gain;
            seeds.push_back(u);
            cout << "[" << S.size() << "/" << k << "] Pick " << u << "   Δ≈" << gain << endl;
            ++round;
        }
    }

    return seeds;
}

// Main
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <data_dir> [k=20] [p=0.01] [mc=200]\n";
        return 1;
    }

    string data_dir = argv[1];
    int k = (argc >= 3) ? stoi(argv[2]) : 20;
    double p = (argc >= 4) ? stod(argv[3]) : 0.01;
    int mc = (argc >= 5) ? stoi(argv[4]) : 200;

    string edge_file = data_dir + "/twitter_combined.txt";
    cout << "Loading graph..." << endl;
    Graph G = load_graph(edge_file);

    cout << "Running CELF-accelerated greedy IC..." << endl;
    vector<string> seeds = celf_ic(G, k, p, mc);

    cout << "\n→ Final seed set:\n";
    for (size_t i = 0; i < seeds.size(); ++i) {
        cout << " " << (i + 1) << ". " << seeds[i] << endl;
    }

    return 0;
}

