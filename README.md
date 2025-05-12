````markdown
# Parallel Social Behavior-Based Algorithm for Identification of Influential Users in Social Network

This project implements a parallel graph algorithm based on the PSAIIM (Parallel Social Action and Interest-based Influence Maximization) framework, which identifies influential users in large-scale social networks by integrating structural and semantic data. The parallelization is performed using MPI, OpenMP, and METIS for scalable performance on clustered systems.

## Project Overview

Influence maximization is a core task in social network analysis, aiming to identify a set of key users who can maximize information spread. Our implementation leverages the PSAIIM algorithm, which:

- Uses **user interaction behavior** (likes, retweets, mentions) and **interest similarity**.
- Applies **PageRank** for influence power calculation.
- Divides large graphs into communities using **SCC/CAC-based graph partitioning**.
- Implements **parallel BFS-tree-based seed selection**.
- Runs on **MPI + OpenMP** hybrid architecture for efficiency.

### Key Features

- Semantic-aware influence modeling using social behavior and interests.
- Parallel execution using MPI across multiple nodes.
- OpenMP-based intra-node parallelism.
- Graph partitioning with METIS for distributed PageRank computation.
- Evaluation on real-world datasets such as HIGGS Twitter dataset.

## Technologies Used

- **MPI (Message Passing Interface)**
- **OpenMP**
- **METIS** for graph partitioning
- **C++** for implementation
- **Bash/Shell scripts** for deployment
- **SNAP Datasets**

## Dataset

We use the [HIGGS Twitter Dataset](https://snap.stanford.edu/data/higgs-twitter.html), which contains four directed networks describing reply, retweet, and mention activities related to the discovery of the Higgs boson.

## Implementation and Analysis

- Implemented PSAIIM using MPI + OpenMP.
- Partitioned graphs using METIS.
- Evaluated the algorithm on various datasets (HIGGS, Facebook Ego Networks).
- Compared performance: Sequential vs. MPI vs. MPI+OpenMP.
- Performed strong and weak scaling analysis.

## Results Summary

| Metric              | PSAIIM (Parallel) | SAIM (Serial) |
|---------------------|------------------|----------------|
| Execution Time      | ⬇ ~60% faster     | -              |
| Memory Usage        | ⬇ Lower usage     | Higher         |
| Influence Spread    | ⬆ Higher          | Lower          |
| Speedup (8 cores)   | ~5.4x             | 1x             |

## Project Presentation

* [View our Presentation](https://www.canva.com/design/DAGk6ooVsW8/rRNpXsZAVu0xmd2POUL7VQ/edit)

## Reference

* **Paper**: [Parallel social behavior-based algorithm for identification of influential users in social network (PSAIIM)](https://doi.org/10.1007/s10489-021-02203-x)
* **Report**: [Performance Analysis and Report](https://docs.google.com/document/d/1nGHrenu3KWQEKBYHoJRV-0w3UdyZvnYzDjqR9B9va6w/edit)

## Team Members

* Emaan \[emaan\@...]
* Aden \[aden\@...]
* Inamullah Shaikh \[inamullahshaikh\@...]

## How to Run

1. **Set up MPI cluster** on two or more Ubuntu machines using SSH.
2. **Install dependencies**:

   ```bash
   sudo apt install openmpi-bin libopenmpi-dev metis
   ```
3. **Compile and Run**:

   ```bash
   mpic++ -fopenmp -o psaim parallel.cpp
   mpirun -np 8 --hostfile hosts.txt ./psaim
   ```

## Future Work

* Extend implementation to GPU using OpenCL.
* Explore influence tracking across evolving (dynamic) networks.
* Integrate real-time data collection from Twitter API.

---
