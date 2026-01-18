#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include <omp.h>
#include <fstream>
#include "../include/data_loader.h"
#include "../include/kmeans.h"
#include "../include/parallel_kmeans.h"
#include "../include/distributed_kmeans.h"
#include "../include/utils.h"
#include "../include/profiler_utils.h"
#include "../include/old_parallel_kmeans.h"

void runTest() {
    std::cout <<"--- Running Data Generation Test ---" << std::endl;
    int numPoints = 10;
    int dim = 2;
    Dataset data = DataLoader::generateData(numPoints, dim, 0.0, 100.0);

    DataLoader::printData(data, numPoints);
    std::cout << "--- Test Finished! ---" << std::endl;
}

void runKMeansSequential(int repeat = 5) {
    std::cout << "--- Running Sequential K-Means benchmark ---" << std::endl;

    int numPoints = 5000000;
    int dim = 3;
    int k = 10;
    int maxIters = 150;


    Dataset dataTemp = DataLoader::generateData(numPoints, dim, 0.0, 1000.0);

    for (int i = 0; i < repeat; ++i) {
        Dataset data = dataTemp;

        std::cin.get();

        KMeans kmeans(k, maxIters);

        double startCpu = ResourceProfiler::getCPUTime();
        auto startWall = std::chrono::high_resolution_clock::now();

        int iters = kmeans.run(data);

        auto endWall = std::chrono::high_resolution_clock::now();
        double endCpu = ResourceProfiler::getCPUTime();

        std::chrono::duration<double> elapsedWall = endWall - startWall;
        double elapsedCpu = endCpu - startCpu;

        // Calculate utilization (100% = 1 core fully loaded)
        double utilization = (elapsedCpu / elapsedWall.count()) * 100.0;

        std::cout << "Run " << (i + 1) << "/" << repeat
                  << " | Wall: " << std::fixed << std::setprecision(4) << elapsedWall.count() << "s"
                  << " | CPU: " << elapsedCpu << "s"
                  << " | Load: " << std::setprecision(1) << utilization << "%"
                  << " (" << iters << " iters)" << std::endl;
    }
}

void runKMeansParallel(int repeat = 10) {
    std::cout << "--- Running Parallel K-Means (Optimized) benchmark ---" << std::endl;

    int numPoints = 5000000;
    int dim = 3;
    int k = 10;
    int maxIters = 150;


    Dataset dataTemp = DataLoader::generateData(numPoints, dim, 0.0, 1000.0);

    for (int i = 0; i < repeat; ++i) {
        Dataset data = dataTemp;
        ParallelKMeans kmeans(k, maxIters);

        std::cin.get();

        double startCpu = ResourceProfiler::getCPUTime();
        auto startWall = std::chrono::high_resolution_clock::now();

        int iters = kmeans.run(data);

        auto endWall = std::chrono::high_resolution_clock::now();
        double endCpu = ResourceProfiler::getCPUTime();

        std::chrono::duration<double> elapsedWall = endWall - startWall;
        double elapsedCpu = endCpu - startCpu;
        double utilization = (elapsedCpu / elapsedWall.count()) * 100.0;

        std::cout << "Run " << (i + 1) << "/" << repeat
                  << " | Wall: " << std::fixed << std::setprecision(4) << elapsedWall.count() << "s"
                  << " | CPU: " << elapsedCpu << "s"
                  << " | Load: " << std::setprecision(1) << utilization << "%"
                  << " (" << iters << " iters)" << std::endl;
    }
}

void runOldParallelKMeans() {
    std::cout << "--- Running OLD (Naive) Parallel K-Means ---" << std::endl;

    int numPoints = 3000000;
    int dim = 1;
    int k = 2;
    int maxIters = 150;

    Dataset data = DataLoader::generateData(numPoints, dim, 0.0, 1000.0);

    OldParallelKMeans oldKmeans(k, maxIters);

    std::cin.get();

    double startCpu = ResourceProfiler::getCPUTime();
    auto startWall = std::chrono::high_resolution_clock::now();

    int iters = oldKmeans.run(data);

    auto endWall = std::chrono::high_resolution_clock::now();
    double endCpu = ResourceProfiler::getCPUTime();

    std::chrono::duration<double> elapsedWall = endWall - startWall;
    double elapsedCpu = endCpu - startCpu;
    double utilization = (elapsedCpu / elapsedWall.count()) * 100.0;

    std::cout << "\n=== OLD Benchmark Results ===" << std::endl;
    std::cout << "Total Wall Time: " << elapsedWall.count() << " s" << std::endl;
    std::cout << "Total CPU Time:  " << elapsedCpu << " s" << std::endl;
    std::cout << "CPU Utilization: " << utilization << " %" << std::endl;
    std::cout << "Iterations:      " << iters << std::endl;
}

void runKMeansDistributed(int repeat = 10) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int numPoints = 5000000;
    int dim = 3;
    int k = 10;
    int maxIters = 150;

    Dataset dataTemp;

    if (world_rank == 0) {
        std::cout << "--- Running Distributed K-Means (MPI) benchmark ---" << std::endl;
        std::cout << "Nodes: " << world_size << " | Points: " << numPoints << std::endl;
        dataTemp = DataLoader::generateData(numPoints, dim, 0.0, 1000.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < repeat; ++i) {
        Dataset data;
        if (world_rank == 0) {
            data = dataTemp;
        }

        DistributedKMeans mpiKmeans(k, maxIters);

        MPI_Barrier(MPI_COMM_WORLD);

        auto startWall = std::chrono::high_resolution_clock::now();

        double startCpuLocal = ResourceProfiler::getCPUTime();

        int iters = mpiKmeans.run(data);

        double endCpuLocal = ResourceProfiler::getCPUTime();

        MPI_Barrier(MPI_COMM_WORLD);

        auto endWall = std::chrono::high_resolution_clock::now();

        double elapsedCpuLocal = endCpuLocal - startCpuLocal;
        std::chrono::duration<double> elapsedWall = endWall - startWall;

        double totalCpuTimeAllNodes = 0.0;

        MPI_Reduce(&elapsedCpuLocal, &totalCpuTimeAllNodes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            double maxPossibleCpuTime = elapsedWall.count() * world_size;
            double utilization = (totalCpuTimeAllNodes / maxPossibleCpuTime) * 100.0;

            std::cout << "Run " << (i + 1) << "/" << repeat
                      << " | Wall: " << std::fixed << std::setprecision(4) << elapsedWall.count() << "s"
                      << " | Total CPU: " << std::setprecision(4) << totalCpuTimeAllNodes << "s"
                      << " | Cluster Load: " << std::setprecision(1) << utilization << "%"
                      << " (" << iters << " iters)" << std::endl;
        }
    }
}


void runScalabilityAnalysis() {
    std::cout << "--- Running Scalability Analysis (Strong Scaling) ---" << std::endl;

    //Parameters
    int numPoints = 500000;
    int dim = 3;
    int k = 10;
    int maxIters = 150;
    int repeat = 3;

    std::vector<int> threadCount = {1, 2, 3, 4, 6, 8, 12, 16};

    std::cout << "Generating dataset (" << numPoints << " points)..." << std::endl;
    Dataset dataFixed = DataLoader::generateData(numPoints, dim, 0.0, 1000.0);

    std::cout << "\n=== RESULTS CSV FORMAT ===" << std::endl;
    std::cout << "Threads,AvgTime_s,Speedup,Efficiency" << std::endl;

    double timeSequential = 0.0;

    for (int t : threadCount) {
        omp_set_num_threads(t);

        double sumTime = 0.0;

        for (int r = 0; r < repeat; ++r) {
            Dataset data = dataFixed;
            ParallelKMeans kmeans(k, maxIters);

            auto start = std::chrono::high_resolution_clock::now();
            kmeans.run(data);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            sumTime += elapsed.count();
        }

        double avgTime = sumTime / repeat;

        if (t == 1) {
            timeSequential = avgTime;
        }

        double speedup = timeSequential / avgTime;
        double efficiency = speedup / t;

        std::cout << t << ","
                  << std::fixed << std::setprecision(5) << avgTime << ","
                  << std::fixed << std::setprecision(2) << speedup << ","
                  << std::fixed << std::setprecision(2) << efficiency << std::endl;
    }
    std::cout << "=== END CSV ===" << std::endl;
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

void runEmpiricalAnalysis() {
    std::cout << "--- EMPIRICAL ANALYSIS: Testing Time Complexity O(N) ---" << std::endl;
    std::vector<int> test_sizes = {100000, 200000, 400000, 800000, 1600000, 3200000};

    int dim = 3;
    int k = 5;
    int maxIters = 50;

    std::ofstream csvFile("empirical_results.csv");
    csvFile << "N,Time_Seconds\n";

    for (int n : test_sizes) {
        std::cout << "Testing N = " << n << " ... " << std::flush;

        Dataset data = DataLoader::generateData(n, dim, 0.0, 1000.0);

        ParallelKMeans kmeans(k, maxIters);

        auto start = std::chrono::high_resolution_clock::now();

        kmeans.run(data);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Done! Time: " << elapsed.count() << "s" << std::endl;
        csvFile << n << "," << elapsed.count() << "\n";
    }

    csvFile.close();
    std::cout << "Results saved to 'empirical_results.csv'. Run Python script now." << std::endl;
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
                runKMeansSequential(1);
            }
        } else if (mode == "--test") {
            if (rank == 0) runTest();
        } else if (mode == "--omp") {
            if (rank == 0) {
                std::cout << "Running parallel OpenMP version..." << std::endl;
                runKMeansParallel(1);
            }
        } else if (mode == "--old") {
            if (rank == 0) runOldParallelKMeans();
        } else if (mode == "--compare") {
            runComparison();
        } else if (mode == "--empirical") {
            runEmpiricalAnalysis();
        } else if (mode == "--scale") {
            if (rank == 0) runScalabilityAnalysis();
        } else if (mode == "--mpi") {
            if (rank == 0) std::cout << "Running distributed MPI version..." << std::endl;
            runKMeansDistributed(1);
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
