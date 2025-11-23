#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include "../include/data_loader.h"
#include "../include/kmeans.h"
#include "../include/utils.h"

void runTest() {
    std::cout <<"--- Running Data Generation Test ---" << std::endl;
    int numPoints = 10;
    int dim = 2;
    Dataset data = DataLoader::generateData(numPoints, dim, 0.0, 100.0);

    DataLoader::printData(data, numPoints);
    std::cout << "--- Test Finished! ---" << std::endl;
}

void runKMeansSequential(int repeat = 10) {
    std::cout << "--- Running Sequential K-Means benchmark ---" << std::endl;
    std::cout << "Executing " << repeat << " runs to average results..." << std::endl;

    //Sim params
    int numPoints = 100000;
    int dim = 3;
    int k = 5;
    int maxIters = 150;

    Dataset dataTemp = DataLoader::generateData(numPoints, dim, 0.0, 1000.0);

    std::vector<double> totalTimes;
    std::vector<double> timePerIter;
    std::vector<int> totalIters;

    for (int i = 0; i < repeat; ++i) {
        Dataset data = dataTemp;

        KMeans kmeans(k, maxIters);

        auto start = std::chrono::high_resolution_clock::now();

        int iters = kmeans.run(data);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        totalTimes.push_back(elapsed.count());
        totalIters.push_back(iters);

        if(iters > 0)
            timePerIter.push_back(elapsed.count() / iters);

        std::cout << "Run " << (i + 1) << "/" << repeat
                  << ": " << std::fixed << std::setprecision(4) << elapsed.count() << "s "
                  << "(" << iters << " iters)" << std::endl;
    }

    // Calculate metrics
    double avgTime = std::accumulate(totalTimes.begin(), totalTimes.end(), 0.0) / repeat;
    double avgIters = std::accumulate(totalIters.begin(), totalIters.end(), 0.0) / (double)repeat;
    double avgTimePerIter = std::accumulate(timePerIter.begin(), timePerIter.end(), 0.0) / static_cast<double>(timePerIter.size());

    double minTime = *std::min_element(totalTimes.begin(), totalTimes.end());
    double maxTime = *std::max_element(totalTimes.begin(), totalTimes.end());

    std::cout << "\n=== Sequential Benchmark Results (" << repeat << " runs) ===" << std::endl;
    std::cout << "Total Time (Avg): " << avgTime << " s" << std::endl;
    std::cout << "Total Time (Min): " << minTime << " s" << std::endl;
    std::cout << "Total Time (Max): " << maxTime << " s" << std::endl;
    std::cout << "Avg Iterations:   " << avgIters << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "AVG TIME PER ITERATION: " << avgTimePerIter << " s" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "(Use 'Avg Time Per Iteration' for comparing OpenMP speedup)" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "==============================" << std::endl;
    std::cout << "   K-Means HPC Project Demo   " << std::endl;
    std::cout << "==============================" << std::endl;

    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "--seq") {
            std::cout << "Running sequential version..." << std::endl;
            runKMeansSequential(10);
        } else if (mode == "--test") {
          runTest();
        } else if (mode == "--omp") {
            std::cout << "Running parallel OpenMP version..." << std::endl;
            // TODO: call runKMeansParallel();
        } else if (mode == "--mpi") {
            std::cout << "Running distributed MPI version..." << std::endl;
            // TODO: call runKMeansDistributed();
        } else {
            std::cout << "Unknown argument. Use one of: --seq, --omp, --mpi" << std::endl;
        }
    } else {
        std::cout << "No mode specified. Defaulting to --seq." << std::endl;
        runKMeansSequential();
    }

    std::cout << "Execution finished successfully!" << std::endl;
    return 0;
}
