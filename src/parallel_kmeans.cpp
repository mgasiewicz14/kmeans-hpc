#include "../include/parallel_kmeans.h"
#include <limits>
#include <random>
#include <iostream>
#include <algorithm>
#include <omp.h>

ParallelKMeans::ParallelKMeans(int k, int maxIter, double threshold)
    : k(k), maxIter(maxIter), threshold(threshold) {}

void ParallelKMeans::initializeCentroids(const Dataset &data) {
    // Initialization can be serial as it's fast and done once
    std::cout << "Initializing centroids (Parallel)..." << std::endl;
    centroids.clear();

    if (data.size() < static_cast<size_t>(k)) {
        std::cerr << "Error: Number of clusters k (" << k << ") is larger than dataset size (" << data.size() << ")." << std::endl;
        return;
    }

    std::vector<size_t> indices(data.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (int i = 0; i < k; ++i) {
        Point p = data[indices[i]];
        centroids.push_back(p);
    }
}

void ParallelKMeans::assignClusters(Dataset& data) {
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        double minDist = std::numeric_limits<double>::max();
        int bestCluster = -1;

        for (int j = 0; j < k; ++j) {
            double dist = distanceSquared(data[i], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = j;
            }
        }
        data[i].clusterId = bestCluster;
    }
}

bool ParallelKMeans::updateCentroids(const Dataset& data) {
    std::vector<Point> newCentroids(k);
    std::vector<int> counts(k, 0);
    size_t dim = data[0].coords.size();

    // Initialize new centroids
    for (int i = 0; i < k; ++i) {
        newCentroids[i].coords.resize(dim, 0.0);
    }

    // Accumulate sums in parallel
    // We need thread-local storage to avoid race conditions on newCentroids and counts
    
    #pragma omp parallel
    {
        std::vector<Point> localCentroids(k);
        std::vector<int> localCounts(k, 0);
        for(int i=0; i<k; ++i) localCentroids[i].coords.resize(dim, 0.0);

        #pragma omp for nowait
        for (int i = 0; i < static_cast<int>(data.size()); ++i) {
            int clusterId = data[i].clusterId;
            if (clusterId == -1) continue;

            localCounts[clusterId]++;
            for (size_t d = 0; d < dim; ++d) {
                localCentroids[clusterId].coords[d] += data[i].coords[d];
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < k; ++i) {
                counts[i] += localCounts[i];
                for (size_t d = 0; d < dim; ++d) {
                    newCentroids[i].coords[d] += localCentroids[i].coords[d];
                }
            }
        }
    }

    // Division by the number of points (serial is fine here, k is small)
    double maxShift = 0.0;
    for (int i = 0; i < k; ++i) {
        if (counts[i] == 0) {
            newCentroids[i] = centroids[i];
            continue;
        }

        for (size_t d = 0; d < dim; ++d) {
            newCentroids[i].coords[d] /= static_cast<double>(counts[i]);
        }

        double shift = distanceSquared(centroids[i], newCentroids[i]);
        if (shift > maxShift) {
            maxShift = shift;
        }
    }

    centroids = newCentroids;
    return maxShift < (threshold * threshold);
}

int ParallelKMeans::run(Dataset& data) {
    if (data.empty() || k <= 0) {
        std::cerr << "Invalid data or k parameter." << std::endl;
        return 0;
    }
    if (data.size() < static_cast<size_t>(k)) {
        std::cerr << "Error: Number of clusters k (" << k << ") is larger than dataset size (" << data.size() << ")." << std::endl;
        return 0;
    }

    initializeCentroids(data);

    int iter = 0;
    bool converged = false;

    while (iter < maxIter && !converged) {
        assignClusters(data);
        converged = updateCentroids(data);
        iter++;
    }

    return iter;
}
