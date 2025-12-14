#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include "../include/data_loader.h"
#include "../include/kmeans.h"
#include "../include/parallel_kmeans.h"
#include "../include/distributed_kmeans.h"
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
    int numPoints = 200000;
    int dim = 3;
    int k = 5;
    int maxIters = 200;

    if (numPoints < k) {
        std::cerr << "Error: numPoints (" << numPoints << ") must be >= k (" << k << ")" << std::endl;
        return;
    }

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

void runKMeansParallel(int repeat = 10) {
    std::cout << "--- Running Parallel K-Means (OpenMP) benchmark ---" << std::endl;
    std::cout << "Executing " << repeat << " runs to average results..." << std::endl;

    //Sim params
    int numPoints = 100000;
    int dim = 3;
    int k = 5;
    int maxIters = 150;

    if (numPoints < k) {
        std::cerr << "Error: numPoints (" << numPoints << ") must be >= k (" << k << ")" << std::endl;
        return;
    }

    Dataset dataTemp = DataLoader::generateData(numPoints, dim, 0.0, 1000.0);

    std::vector<double> totalTimes;
    std::vector<double> timePerIter;
    std::vector<int> totalIters;

    for (int i = 0; i < repeat; ++i) {
        Dataset data = dataTemp;

        ParallelKMeans kmeans(k, maxIters);

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

    std::cout << "\n=== Parallel Benchmark Results (" << repeat << " runs) ===" << std::endl;
    std::cout << "Total Time (Avg): " << avgTime << " s" << std::endl;
    std::cout << "Total Time (Min): " << minTime << " s" << std::endl;
    std::cout << "Total Time (Max): " << maxTime << " s" << std::endl;
    std::cout << "Avg Iterations:   " << avgIters << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "AVG TIME PER ITERATION: " << avgTimePerIter << " s" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
}

void runKMeansDistributed(int repeat = 10) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "--- Running Distributed K-Means (MPI) benchmark ---" << std::endl;
        std::cout << "Executing " << repeat << " runs to average results..." << std::endl;
    }

    //Simulation parameters
    int numPoints = 100000;
    int dim = 3;
    int k = 5;
    int maxIters = 150;

    Dataset dataTemp;
    if (rank == 0) {
        dataTemp = DataLoader::generateData(numPoints, dim, 0.0, 1000.0);
    }

    std::vector<double> totalTimes;
    std::vector<double> timePerIter;
    std::vector<int> totalIters;

    for (int i = 0; i < repeat; ++i) {
        Dataset data;

        if (rank == 0) {
            data = dataTemp;
        }

        DistributedKMeans mpiKmeans(k, maxIters);

        //Synchronization before the clock stats counting time
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();

        int iters = mpiKmeans.run(data);

        //Wait until each process finishes calculating
        MPI_Barrier(MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();

        if (rank == 0) {
            std::chrono::duration<double> elapsed = end - start;

            totalTimes.push_back(elapsed.count());
            totalIters.push_back(iters);

            if (iters > 0)
                timePerIter.push_back(elapsed.count() / iters);

            std::cout << "Run " << (i + 1) << "/" << repeat
                      << ": " << std::fixed << std::setprecision(4) << elapsed.count() << "s "
                      << "(" << iters << " iters)" << std::endl;
        }
    }

    //Calculate metrics (master only)
    if (rank == 0) {
        double avgTime = std::accumulate(totalTimes.begin(), totalTimes.end(), 0.0) / repeat;
        double avgIters = std::accumulate(totalIters.begin(), totalIters.end(), 0.0) / (double)repeat;
        double avgTimePerIter = std::accumulate(timePerIter.begin(), timePerIter.end(), 0.0) / static_cast<double>(timePerIter.size());

        double minTime = *std::min_element(totalTimes.begin(), totalTimes.end());
        double maxTime = *std::max_element(totalTimes.begin(), totalTimes.end());

        std::cout << "\n=== Distributed Benchmark Results (" << repeat << " runs) ===" << std::endl;
        std::cout << "Total Time (Avg): " << avgTime << " s" << std::endl;
        std::cout << "Total Time (Min): " << minTime << " s" << std::endl;
        std::cout << "Total Time (Max): " << maxTime << " s" << std::endl;
        std::cout << "Avg Iterations:   " << avgIters << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "AVG TIME PER ITERATION: " << avgTimePerIter << " s" << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
    }
}

void runComparison() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numPoints = 200000;
    int dim = 3;
    int k = 5;
    int maxIters = 300;

    double timeSeq = 0.0; int iterSeq = 0;
    double timePar = 0.0; int iterPar = 0;
    double timeDist = 0.0; int iterDist = 0;

    Dataset dataOriginal;

    if (rank == 0) {
        std::cout << "--- Running Full Comparison (Seq vs Parallel vs Distributed) ---" << std::endl;

        if (numPoints < k) {
            std::cerr << "Error: numPoints < k" << std::endl;
        }

        std::cout << "Generating dataset (" << numPoints << " points, " << dim << " dims)..." << std::endl;
        dataOriginal = DataLoader::generateData(numPoints, dim, 0.0, 100.0);
    }

    if (rank == 0) {
        std::cout << "\n1. Sequential K-Means..." << std::endl;
        Dataset dataSeq = dataOriginal; // Kopia dla czystego startu

        KMeans seq(k, maxIters);
        auto start = std::chrono::high_resolution_clock::now();
        iterSeq = seq.run(dataSeq);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        timeSeq = elapsed.count();
        std::cout << "   Time: " << timeSeq << "s, Iters: " << iterSeq << std::endl;
    }

    if (rank == 0) {
        std::cout << "\n2. Parallel K-Means (OpenMP)..." << std::endl;
        Dataset dataPar = dataOriginal; // Kopia

        ParallelKMeans par(k, maxIters);
        auto start = std::chrono::high_resolution_clock::now();
        iterPar = par.run(dataPar);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        timePar = elapsed.count();
        std::cout << "   Time: " << timePar << "s, Iters: " << iterPar << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::cout << "\n3. Distributed K-Means (MPI)..." << std::endl;

    Dataset dataDist;
    if (rank == 0) {
        dataDist = dataOriginal;
    }

    DistributedKMeans dist(k, maxIters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto startDistTime = std::chrono::high_resolution_clock::now();

    iterDist = dist.run(dataDist);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDistTime = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::chrono::duration<double> elapsed = endDistTime - startDistTime;
        timeDist = elapsed.count();
        std::cout << "   Time: " << timeDist << "s, Iters: " << iterDist << std::endl;
    }

    if (rank == 0) {
        std::cout << "\n=== Final Comparison Results ===" << std::endl;
        std::cout << "Sequential Time: " << std::fixed << std::setprecision(4) << timeSeq << " s" << std::endl;
        std::cout << "Parallel Time:   " << timePar << " s" << std::endl;
        std::cout << "Distributed Time:" << timeDist << " s" << std::endl;

        std::cout << "--------------------------------" << std::endl;

        double speedupPar = timeSeq / timePar;
        double speedupDist = timeSeq / timeDist;

        std::cout << "Speedup Parallel vs Seq:    " << std::fixed << std::setprecision(2) << speedupPar << "x" << std::endl;
        std::cout << "Speedup Distributed vs Seq: " << speedupDist << "x" << std::endl;

        std::cout << "--------------------------------" << std::endl;

        if (iterSeq == iterPar && iterSeq == iterDist) {
            std::cout << "SUCCESS: All versions finished in " << iterSeq << " iterations." << std::endl;
        } else {
            std::cout << "NOTE: Iterations differ (Seq:" << iterSeq << ", Par:" << iterPar << ", Dist:" << iterDist << ")." << std::endl;
            std::cout << "      (This is normal due to random initialization differences if seeds are not synced across versions)" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "==============================" << std::endl;
        std::cout << "   K-Means HPC Project Demo   " << std::endl;
        std::cout << "==============================" << std::endl;
    }

    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "--seq") {
            if (rank == 0) {
                std::cout << "Running sequential version..." << std::endl;
                runKMeansSequential(10);
            }
        } else if (mode == "--test") {
            if (rank == 0) runTest();
        } else if (mode == "--omp") {
            if (rank == 0) {
                std::cout << "Running parallel OpenMP version..." << std::endl;
                runKMeansParallel(10);
            }
        } else if (mode == "--compare") {
            runComparison();
        } else if (mode == "--mpi") {
            if (rank == 0) std::cout << "Running distributed MPI version..." << std::endl;
            runKMeansDistributed(10);
        } else {
            if (rank == 0) std::cout << "Unknown argument. Use one of: --seq, --omp, --compare, --mpi" << std::endl;
        }
    } else {
        if (rank == 0) {
            std::cout << "No mode specified. Defaulting to --seq." << std::endl;
            runKMeansSequential();
        }
    }

    if (rank == 0) std::cout << "Execution finished successfully!" << std::endl;

    MPI_Finalize();
    return 0;
}
