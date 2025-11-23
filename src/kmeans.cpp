#include "../include/kmeans.h"
#include <limits>
#include <random>
#include <iostream>
#include <algorithm>

KMeans::KMeans(int k, int maxIter, double threshold)
    : k(k), maxIter(maxIter), threshold(threshold) {}

void KMeans::initializeCentroids(const Dataset &data) {
    std::cout << "Initializing centroids..." << std::endl;
    centroids.clear();
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
            newCentroids[i].coords[d] /= counts[i];
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