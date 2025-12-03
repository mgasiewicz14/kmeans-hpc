#include "../include/distributed_kmeans.h"
#include <iostream>
#include <vector>
#include <limits>
#include <random>
#include <algorithm>

DistributedKMeans::DistributedKMeans(int k, int maxIter, double threshold)
        : k(k), maxIter(maxIter), threshold(threshold) {
    // Download MPI env data
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}

DistributedKMeans::~DistributedKMeans() {}

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
    int n_points = 0;
    int dim = 0;

    //Data distribution

    if (world_rank == 0) {
        if (!data.empty()) {
            n_points = static_cast<int>(data.size());
            dim = static_cast<int>(data[0].coords.size());
            initializeCentroids(data);
        }
    }

    // MPI_Bcast(buffer, count, datatype, root, communicator)
    MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //Calculate how much points will every process get
    std::vector<int> send_counts(world_size);
    std::vector<int> displs(world_size);

    int base_count = n_points / world_size;
    int remainder = n_points % world_size;

    for (int i = 0; i < world_size; ++i) {
        send_counts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
    }

    int local_n = send_counts[world_rank];

    //MPI operates on double arrays
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

    //Send data
    MPI_Scatterv(
            global_flat_data.data(), send_counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, // Send (Rank 0)
            local_flat_data.data(), local_n * dim, MPI_DOUBLE,                                      // Recv (All ranks)
            0, MPI_COMM_WORLD
    );

    Dataset local_data(local_n);
    for (int i = 0; i < local_n; ++i) {
        std::vector<double> coords(dim);
        for (int d = 0; d < dim; ++d) {
            coords[d] = local_flat_data[i * dim + d];
        }
        local_data[i] = Point(coords);
    }

    //Main loop
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

    while (iter < maxIter && !converged) {
        MPI_Bcast(flat_centroids.data(), k * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (world_rank != 0) {
            for (int i = 0; i < k; ++i) {
                for (int d = 0; d < dim; ++d) {
                    centroids[i].coords[d] = flat_centroids[i * dim + d];
                }
            }
        }

        std::vector<double> local_sums(k * dim, 0.0);
        std::vector<int> local_counts(k, 0);

        for (auto& p : local_data) {
            double minDist = std::numeric_limits<double>::max();
            int bestCluster = -1;

            for (int j = 0; j < k; ++j) {
                double dist = distanceSquared(p, centroids[j]);
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

        //Global reduction
        std::vector<double> global_sums(k * dim);
        std::vector<int> global_counts(k);

        MPI_Allreduce(local_sums.data(), global_sums.data(), k * dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        //Update the centroids
        double maxShift = 0.0;
        for (int i = 0; i < k; ++i) {
            if (global_counts[i] == 0) {
                continue;
            }

            Point newCentroid;
            newCentroid.coords.resize(dim);

            for (int d = 0; d < dim; ++d) {
                newCentroid.coords[d] = global_sums[i * dim + d] / global_counts[i];
            }

            double shift = distanceSquared(centroids[i], newCentroid);
            if (shift > maxShift) maxShift = shift;

            centroids[i] = newCentroid;
            for(int d=0; d<dim; ++d) {
                flat_centroids[i * dim + d] = newCentroid.coords[d];
            }
        }

        if (maxShift < threshold * threshold) {
            converged = true;
        }
        iter++;
    }

    return iter;
}