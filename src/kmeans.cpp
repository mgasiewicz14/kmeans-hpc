#include "../include/kmeans.h"
#include <chrono>
#include <limits>
#include <random>
#include <iostream>
#include <algorithm>

KMeans::KMeans(int k, int maxIter, double threshold)
    : k(k), maxIter(maxIter), threshold(threshold) {}

void KMeans::initializeCentroids(const Dataset &data) {
    std::cout << "Initializing centroids..." << std::endl;
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
//Assign every point to the nearest centroid
void KMeans::assignClusters(Dataset& data) {
    for (auto& point : data) {
        double minDist = std::numeric_limits<double>::max();
        int bestCluster = -1;

        for (int i = 0; i < k; ++i) {
            double dist = distanceSquared(point, centroids[i]);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = i;
            }
        }
        point.clusterId = bestCluster;
    }
}
//Returns true if the algorithm has reached convergence.
bool KMeans::updateCentroids(const Dataset& data) {
    std::vector<Point> newCentroids(k);
    std::vector<int> counts(k, 0);

    // Initialize new centroids with zeros
    size_t dim = data[0].coords.size();
    for (int i = 0; i < k; ++i) {
        newCentroids[i].coords.resize(dim, 0.0);
    }

    // Summing the coords of points in every cluster
    for (const auto& point : data) {
        int clusterId = point.clusterId;
        if (clusterId == -1) continue;

        counts[clusterId]++;
        for (size_t d = 0; d < dim; ++d) {
            newCentroids[clusterId].coords[d] += point.coords[d];
        }
    }

    // Division by the number of points
    double maxShift = 0.0;
    for (int i = 0; i < k; ++i) {
        if (counts[i] == 0) {
            newCentroids[i] = centroids[i];
            continue;
        }

        for (size_t d = 0; d < dim; ++d) {
            newCentroids[i].coords[d] /= static_cast<double>(counts[i]);
        }

        //Check how far the centroid has shifted
        double shift = distanceSquared(centroids[i], newCentroids[i]);
        if (shift > maxShift) {
            maxShift = shift;
        }
    }

    centroids = newCentroids;

    // Check the convergence (squared th, because of the squared value of distance)
    return maxShift < (threshold * threshold);
}

int KMeans::run(Dataset& data) {
    if (data.empty() || k <= 0) {
        std::cerr << "Invalid data or k parameter." << std::endl;
        return 0;
    }
    if (data.size() < static_cast<size_t>(k)) {
        std::cerr << "Error: Number of clusters k (" << k << ") is larger than dataset size (" << data.size() << ")." << std::endl;
        return 0;
    }

    initTime = 0.0;
    totalAssignTime = 0.0;
    totalUpdateTime = 0.0;

    auto startInit = std::chrono::high_resolution_clock::now();

    initializeCentroids(data);

    auto endInit = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diffInit = endInit - startInit;
    initTime = diffInit.count();

    int iter = 0;
    bool converged = false;

    while (iter < maxIter && !converged) {

        auto startAssign = std::chrono::high_resolution_clock::now();
        assignClusters(data);
        auto endAssign = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffAssign = endAssign - startAssign;
        totalAssignTime += diffAssign.count();

        auto startUpdate = std::chrono::high_resolution_clock::now();
        converged = updateCentroids(data);
        auto endUpdate = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diffUpdate = endUpdate - startUpdate;
        totalUpdateTime += diffUpdate.count();

        iter++;
    }

    double totalTotalTime = initTime + totalAssignTime + totalUpdateTime;

    std::cout << "\n========================================" << std::endl;
    std::cout << "      DETAILED PROFILING REPORT" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total Iterations: " << iter << std::endl;
    std::cout << "Total Wall Time:  " << totalTotalTime << " s" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "1. Initialization:      " << initTime << " s ("
              << (initTime / totalTotalTime) * 100.0 << "%)" << std::endl;

    std::cout << "2. AssignClusters (O(N)): " << totalAssignTime << " s ("
              << (totalAssignTime / totalTotalTime) * 100.0 << "%)" << std::endl;

    std::cout << "3. UpdateCentroids (O(N)): " << totalUpdateTime << " s ("
              << (totalUpdateTime / totalTotalTime) * 100.0 << "%)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return iter;
}