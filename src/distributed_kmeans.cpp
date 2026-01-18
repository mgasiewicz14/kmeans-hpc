#include "../include/distributed_kmeans.h"
#include <iostream>
#include <vector>
#include <limits>
#include <random>
#include <algorithm>
#include <iomanip>
#include <fstream>


DistributedKMeans::DistributedKMeans(int k, int maxIter, double threshold)
        : k(k), maxIter(maxIter), threshold(threshold) {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}

DistributedKMeans::~DistributedKMeans() {}


void DistributedKMeans::addLog(double start, double end, int type, const std::string& name) {
    logs.push_back({world_rank, start, end, (EventType)type, name});
}

void DistributedKMeans::saveLogsToCSV() {
    std::string fName = "mpi_log_rank_" + std::to_string(world_rank) + ".csv";
    std::ofstream file(fName);
    file << "Rank,Start,End,Type,Name\n";
    for (const auto& log : logs) {
        file << log.rank << ","
             << std::fixed << std::setprecision(6) << log.start << ","
             << log.end << ","
             << (log.type == COMP ? "COMP" : "COMM") << ","
             << log.name << "\n";
    }
    file.close();
    if (world_rank == 0) std::cout << "Logs saved to " << fName << " (and others)" << std::endl;
}

void DistributedKMeans::initializeCentroids(const Dataset& data) {
    if (world_rank == 0) {
        std::cout << "[MPI Rank 0] Initializing centroids..." << std::endl;
        centroids.clear();
        std::vector<size_t> indices(data.size());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        for (int i = 0; i < k; ++i) {
            centroids.push_back(data[indices[i]]);
        }
    }
}

int DistributedKMeans::run(Dataset& data) {
    logs.clear();

    int n_points = 0;
    int dim = 0;

    // Data setup
    double t_start = MPI_Wtime();

    if (world_rank == 0) {
        if (!data.empty()) {
            n_points = static_cast<int>(data.size());
            dim = static_cast<int>(data[0].coords.size());
            initializeCentroids(data);
        }
    }
    // addLog(t_start, MPI_Wtime(), COMP, "InitLocal");

    // Metadata transmission
    double t_comm = MPI_Wtime();
    MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    addLog(t_comm, MPI_Wtime(), COMM, "MetaBcast");

    std::vector<int> send_counts(world_size);
    std::vector<int> displs(world_size);

    int base_count = n_points / world_size;
    int remainder = n_points % world_size;

    for (int i = 0; i < world_size; ++i) {
        send_counts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
    }

    int local_n = send_counts[world_rank];

    // Prepare the buffers
    std::vector<double> global_flat_data;
    std::vector<double> local_flat_data(local_n * dim);

    if (world_rank == 0) {
        global_flat_data.resize(n_points * dim);
        for (int i = 0; i < n_points; ++i) {
            for (int d = 0; d < dim; ++d) {
                global_flat_data[i * dim + d] = data[i].coords[d];
            }
        }
    }

    std::vector<int> send_counts_doubles(world_size);
    std::vector<int> displs_doubles(world_size);
    for(int i=0; i<world_size; ++i) {
        send_counts_doubles[i] = send_counts[i] * dim;
        displs_doubles[i] = displs[i] * dim;
    }

    // Sending data
    t_comm = MPI_Wtime();

    double* sendbuf = (world_rank == 0) ? global_flat_data.data() : nullptr;
    MPI_Scatterv(
            sendbuf, send_counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE,
            local_flat_data.data(), local_n * dim, MPI_DOUBLE,
            0, MPI_COMM_WORLD
    );
    addLog(t_comm, MPI_Wtime(), COMM, "ScatterData");

    // Creating Point objects
    double t_comp = MPI_Wtime();
    Dataset local_data(local_n);
    for (int i = 0; i < local_n; ++i) {
        std::vector<double> coords(dim);
        for (int d = 0; d < dim; ++d) {
            coords[d] = local_flat_data[i * dim + d];
        }
        local_data[i] = Point(coords);
    }
    addLog(t_comp, MPI_Wtime(), COMP, "RebuildData");

    //Main loop setup
    std::vector<double> flat_centroids(k * dim);
    if (world_rank == 0) {
        for (int i = 0; i < k; ++i) {
            for (int d = 0; d < dim; ++d) {
                flat_centroids[i * dim + d] = centroids[i].coords[d];
            }
        }
    } else {
        centroids.resize(k);
        for(int i=0; i<k; ++i) centroids[i].coords.resize(dim);
    }

    int iter = 0;
    bool converged = false;

    // Main loop
    while (iter < maxIter && !converged) {
        // Bcast Centroids (COMM)
        t_comm = MPI_Wtime();
        MPI_Bcast(flat_centroids.data(), k * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        addLog(t_comm, MPI_Wtime(), COMM, "BcastCentr");

        if (world_rank != 0) {
            for (int i = 0; i < k; ++i) {
                for (int d = 0; d < dim; ++d) {
                    centroids[i].coords[d] = flat_centroids[i * dim + d];
                }
            }
        }

        // Local computing
        t_comp = MPI_Wtime();
        std::vector<double> local_sums(k * dim, 0.0);
        std::vector<int> local_counts(k, 0);

        for (auto& p : local_data) {
            double minDist = std::numeric_limits<double>::max();
            int bestCluster = -1;

            for (int j = 0; j < k; ++j) {
                // Inline distance calculation for speed
                double dist = 0.0;
                for(int d=0; d<dim; ++d) {
                    double diff = p.coords[d] - centroids[j].coords[d];
                    dist += diff * diff;
                }

                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }
            p.clusterId = bestCluster;

            local_counts[bestCluster]++;
            for (int d = 0; d < dim; ++d) {
                local_sums[bestCluster * dim + d] += p.coords[d];
            }
        }
        addLog(t_comp, MPI_Wtime(), COMP, "CalcLocal"); // Zielony pasek na wykresie

        // Global reduction
        t_comm = MPI_Wtime();
        std::vector<double> global_sums(k * dim);
        std::vector<int> global_counts(k);

        MPI_Allreduce(local_sums.data(), global_sums.data(), k * dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        addLog(t_comm, MPI_Wtime(), COMM, "AllReduce"); // Czerwony pasek

        // Update
        t_comp = MPI_Wtime();
        double maxShift = 0.0;
        for (int i = 0; i < k; ++i) {
            if (global_counts[i] == 0) continue;

            Point newCentroid;
            newCentroid.coords.resize(dim);

            for (int d = 0; d < dim; ++d) {
                newCentroid.coords[d] = global_sums[i * dim + d] / global_counts[i];
            }

            // Simple distance check
            double shift = 0.0;
            for(int d=0; d<dim; ++d) {
                double diff = centroids[i].coords[d] - newCentroid.coords[d];
                shift += diff * diff;
            }
            if (shift > maxShift) maxShift = shift;

            centroids[i] = newCentroid;
            for(int d=0; d<dim; ++d) {
                flat_centroids[i * dim + d] = newCentroid.coords[d];
            }
        }

        if (maxShift < threshold * threshold) {
            converged = true;
        }
        addLog(t_comp, MPI_Wtime(), COMP, "Update");

        iter++;
    }

    // Save logs
    saveLogsToCSV();
    return iter;
}