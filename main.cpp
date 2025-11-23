#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include "include/kmeans.h"
#include "include/parallel_kmeans.h"
#include "include/utils.h"

// Function to generate random dataset
Dataset generateDataset(int numPoints, int dim, double minVal, double maxVal) {
    Dataset data;
    data.reserve(numPoints);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(minVal, maxVal);

    for (int i = 0; i < numPoints; ++i) {
        Point p;
        p.coords.resize(dim);
        for (int d = 0; d < dim; ++d) {
            p.coords[d] = dis(gen);
        }
        p.clusterId = -1;
        data.push_back(p);
    }
    return data;
}

int main() {
    int numPoints = 100000;
    int dim = 5;
    int k = 10;
    int maxIter = 100;
    
    std::cout << "Generating dataset with " << numPoints << " points, " << dim << " dimensions..." << std::endl;
    
    if (numPoints < k) {
        std::cerr << "Error: numPoints (" << numPoints << ") must be >= k (" << k << ")" << std::endl;
        return 1;
    }

    Dataset data = generateDataset(numPoints, dim, 0.0, 100.0);
    
    // Create copies for both algorithms to ensure fair comparison (same starting data)
    Dataset dataSerial = data;
    Dataset dataParallel = data;

    std::cout << "\n--- Serial K-Means ---" << std::endl;
    KMeans serialKMeans(k, maxIter);
    auto startSerial = std::chrono::high_resolution_clock::now();
    int iterSerial = serialKMeans.run(dataSerial);
    auto endSerial = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSerial = endSerial - startSerial;
    
    std::cout << "Iterations: " << iterSerial << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(4) << durationSerial.count() << " seconds" << std::endl;

    std::cout << "\n--- Parallel K-Means (OpenMP) ---" << std::endl;
    ParallelKMeans parallelKMeans(k, maxIter);
    auto startParallel = std::chrono::high_resolution_clock::now();
    int iterParallel = parallelKMeans.run(dataParallel);
    auto endParallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationParallel = endParallel - startParallel;

    std::cout << "Iterations: " << iterParallel << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(4) << durationParallel.count() << " seconds" << std::endl;

    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "Speedup: " << durationSerial.count() / durationParallel.count() << "x" << std::endl;
    
    return 0;
}