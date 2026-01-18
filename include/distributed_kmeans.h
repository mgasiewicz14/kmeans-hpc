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

    enum EventType { COMP = 0, COMM = 1 };
    struct LogEvent {
        int rank;
        double start;
        double end;
        EventType type;
        std::string name;
    };
    std::vector<LogEvent> logs;
    void addLog(double start, double end, int type, const std::string& name);

    std::vector<Point> centroids;

    void initializeCentroids(const Dataset& data);

public:
    DistributedKMeans(int k, int maxIter = 100, double threshold = 1e-4);
    ~DistributedKMeans();

    int run(Dataset& data);
    void saveLogsToCSV();

    [[nodiscard]] const std::vector<Point>& getCentroids() const {return centroids;}
};