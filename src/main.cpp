#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "==============================" << std::endl;
    std::cout << "   K-Means HPC Project Demo   " << std::endl;
    std::cout << "==============================" << std::endl;

    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "--seq") {
            std::cout << "Running sequential version..." << std::endl;
            // TODO: call runKMeansSequential();
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
        std::cout << "No mode specified. Default: sequential." << std::endl;
        // TODO: call runKMeansSequential();
    }

    std::cout << "Execution finished successfully!" << std::endl;
    return 0;
}
