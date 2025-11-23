#ifndef KMEANS_H
#define KMEANS_H

#include "utils.h"
#include <vector>

class KMeans {
private:
    int k;
    int maxIter;
    double threshold; // Convergence threshold, if changes are smaller, stop the alg

    std::vector<Point> centroids;

    void initializeCentroids(const Dataset& data);
    void assignClusters(Dataset& data);
    bool updateCentroids(const Dataset& data);
public:
    KMeans(int k, int maxIter = 100, double threshold = 1e-4);

    int run(Dataset& data);
    const std::vector<Point>& getCentroids() const {return centroids;}
};

#endif // KMEANS_H