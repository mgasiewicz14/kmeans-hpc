#include <iostream>
#include <string>
#include <vector>
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

int main(int argc, char* argv[]) {
    std::cout << "==============================" << std::endl;
    std::cout << "   K-Means HPC Project Demo   " << std::endl;
    std::cout << "==============================" << std::endl;

    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "--seq") {
            std::cout << "Running sequential version..." << std::endl;
            // TODO: call runKMeansSequential();
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
        std::cout << "No mode specified. Running basic test." << std::endl;
        runTest();
    }

    std::cout << "Execution finished successfully!" << std::endl;
    return 0;
}
