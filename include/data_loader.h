#pragma once

#include "utils.h"
#include <string>
#include <vector>

class DataLoader {
public:
    // Generates random synthetic dataset
    static Dataset generateData(int numPoints, int dim, double minVal, double maxVal);
    // Optional loading data from csv file (todo incase if something doesnt work with synth data generation)
    static Dataset loadFromCSV(const std::string& filename);
    //Function to print fragments of data (used for debugging)
    static void printData(const Dataset& data, int numLines = 5);
};