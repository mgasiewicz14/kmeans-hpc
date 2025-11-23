#pragma once

#include "utils.h"
#include <vector>

class ParallelKMeans {
private:
    int k;
    int maxIter;
    double threshold;

    std::vector<Point> centroids;

    void initializeCentroids(const Dataset& data);
    void assignClusters(Dataset& data);
    bool updateCentroids(const Dataset& data);

public:
    ParallelKMeans(int k, int maxIter = 100, double threshold = 1e-4);

    int run(Dataset& data);
    [[nodiscard]] const std::vector<Point>& getCentroids() const { return centroids; }
};

