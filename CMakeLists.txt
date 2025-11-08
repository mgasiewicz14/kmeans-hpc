cmake_minimum_required(VERSION 3.12)
project(KMeans_HPC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    add_definitions(-DUSE_OPENMP)
endif()

set(MPI_CXX_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
set(MPIEXEC_EXECUTABLE "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")

# Enable MPI
find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    message(STATUS "MPI found")
    add_definitions(-DUSE_MPI)
endif()

# Source files
file(GLOB SRC_FILES src/*.cpp)

# Executable
add_executable(kmeans_hpc ${SRC_FILES})

# Include directories
target_include_directories(kmeans_hpc PRIVATE include ${MPI_INCLUDE_PATH})

# Link libraries
target_link_libraries(kmeans_hpc PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
