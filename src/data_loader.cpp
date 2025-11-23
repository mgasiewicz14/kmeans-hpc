#include "../include/data_loader.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

Dataset DataLoader::generateData(int numPoints, int dim, double minVal, double maxVal) {
    std::cout << "Generating " << numPoints << " points in " << dim << " dimensions..." << std::endl;

    Dataset data;
    data.reserve(numPoints); //Memory reservation

    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister gen
    std::uniform_real_distribution<> dis(minVal, maxVal); // Uniform Distribution

    for (int i = 0; i < numPoints; i++){
        std::vector<double> coords(dim);
        for (int d = 0; d < dim; d++) {
            coords[d] = dis(gen);
        }
        data.emplace_back(coords);
    }

    std::cout << "Generation complete!" <<std::endl;
    return data;
}

void DataLoader::printData(const Dataset& data, int numLines) {
    int limit = std::min((int)data.size(), numLines);
    std::cout << "--- Data Sample (First " << limit << " points) ---" << std::endl;
    for (int i = 0; i < limit; ++i) {
        std::cout << "P" << i << ": [";
        for (size_t d = 0; d < data[i].coords.size(); ++d) {
            std::cout << std::fixed << std::setprecision(2) << data[i].coords[d] << (d < data[i].coords.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "-------------------------------------------" << std::endl;
}

//Placeholder for CSV Dataloader
Dataset DataLoader::loadFromCSV(const std::string& filename) {
    std::cout << "[DataLoader] Loading from CSV is not fully implemented yet." << std::endl;
    return Dataset();
}
