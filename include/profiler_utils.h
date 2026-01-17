#pragma once
#ifdef _WIN32
#include <windows.h>
#else
#include <ctime>
#endif
#include <iostream>

class ResourceProfiler {
public:
    // Returns the total CPU time (User + Kernel) in seconds.
    static double getCPUTime() {
#ifdef _WIN32
        FILETIME a, b, c, d;
        if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
            unsigned long long kernel = ((unsigned long long)c.dwHighDateTime << 32) | c.dwLowDateTime;
            unsigned long long user   = ((unsigned long long)d.dwHighDateTime << 32) | d.dwLowDateTime;
            return (double)(kernel + user) * 1e-7;
        }
        return 0.0;
#else
        return (double)clock() / CLOCKS_PER_SEC;
#endif
    }
};