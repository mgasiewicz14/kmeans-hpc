#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

struct Point {
    std::vector<double> coords;
    int clusterId;

    //Default constructor
    Point() : clusterId(-1){}
    //Data constructor
    explicit Point(const std::vector<double>& c) : coords(c), clusterId(-1){}
};

using Dataset = std::vector<Point>;


//Function for distance calculation
[[nodiscard]] inline double distanceSquared(const Point& p1, const Point& p2){
    assert(p1.coords.size() == p2.coords.size() && "Point dimensions must match");
    double sum = 0.0;
    size_t size = p1.coords.size();
    for (size_t i = 0; i < size; i++){
        double diff = p1.coords[i] - p2.coords[i];
        sum += diff * diff;
    }
    return sum;
}