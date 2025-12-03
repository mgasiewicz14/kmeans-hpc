#pragma once

#include "utils.h"
#include <vector>
#include <mpi.h>

class DistributedKMeans {
private:
    int k;
    int maxIter;
    double threshold;

    //MPI data
    int world_rank; // process ID
    int world_size; // number of processes

    std::vector<Point> centroids;

    void initializeCentroids(const Dataset& data);

public:
    DistributedKMeans(int k, int maxIter = 100, double threshold = 1e-4);
    ~DistributedKMeans();

    int run(Dataset& data);

    [[nodiscard]] const std::vector<Point>& getCentroids() const {return centroids;}
};