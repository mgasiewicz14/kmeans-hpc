#include "../include/old_parallel_kmeans.h"
#include <omp.h>
#include <random>
#include <algorithm>
#include <iostream>

OldParallelKMeans::OldParallelKMeans(int k, int maxIter, double threshold)
        : k(k), maxIter(maxIter), threshold(threshold) {}

void OldParallelKMeans::initializeCentroids(const Dataset &data) {
    centroids.clear();
    std::vector<size_t> indices(data.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
    std::random_device rd; std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    for (int i = 0; i < k; ++i) centroids.push_back(data[indices[i]]);
}

void OldParallelKMeans::assignClusters(Dataset& data) {
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        double minDist = 1e30;
        int bestCluster = -1;
        for (int j = 0; j < k; ++j) {
            double dist = distanceSquared(data[i], centroids[j]);
            if (dist < minDist) { minDist = dist; bestCluster = j; }
        }
        data[i].clusterId = bestCluster;
    }
}

bool OldParallelKMeans::updateCentroids(const Dataset& data) {
    std::vector<Point> newCentroids(k);
    std::vector<int> counts(k, 0);
    size_t dim = data[0].coords.size();

    for(int i=0; i<k; ++i) newCentroids[i].coords.resize(dim, 0.0);


    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        int clusterId = data[i].clusterId;
        if (clusterId == -1) continue;

        #pragma omp atomic
        counts[clusterId]++;

        for (size_t d = 0; d < dim; ++d) {
            #pragma omp atomic
            newCentroids[clusterId].coords[d] += data[i].coords[d];
        }
    }

    double maxShift = 0.0;
    for (int i = 0; i < k; ++i) {
        if (counts[i] == 0) { newCentroids[i] = centroids[i]; continue; }
        for (size_t d = 0; d < dim; ++d) newCentroids[i].coords[d] /= counts[i];
        double shift = distanceSquared(centroids[i], newCentroids[i]);
        if (shift > maxShift) maxShift = shift;
    }
    centroids = newCentroids;
    return maxShift < (threshold * threshold);
}

int OldParallelKMeans::run(Dataset& data) {
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