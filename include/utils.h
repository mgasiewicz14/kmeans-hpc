#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cmath>
#include <iostream>

struct Point {
    std::vector<double> coords;
    int clusterId;

    //Default constructor
    Point() : clusterId(-1){}
    //Data constructor
    Point(const std::vector<double>& c) : coords(c), clusterId(-1){}
};

using Dataset = std::vector<Point>;


//Function for distance calculation
inline double distanceSquared(const Point& p1, const Point& p2){
    double sum = 0.0;
    for (size_t i = 0; i < p1.coords.size(); i++){
        double diff = p1.coords[i] - p2.coords[i];
        sum += diff * diff;
    }
    return sum;
}

#endif // UTILS_H