cmake_minimum_required(VERSION 3.12)
project(KMeans_HPC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# --- OPENMP CONFIGURATION ---
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Version: ${OpenMP_CXX_VERSION}")
    # To jest ważne dla MinGW:
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DUSE_OPENMP)
endif()

# --- MPI CONFIGURATION ---
# Ścieżki do MS-MPI (Microsoft MPI)
set(MPI_CXX_INCLUDE_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")
set(MPIEXEC_EXECUTABLE "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")

find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    message(STATUS "MPI found")
    add_definitions(-DUSE_MPI)
endif()

# --- SOURCE FILES ---
file(GLOB SRC_FILES src/*.cpp)

# --- EXECUTABLE ---
add_executable(kmeans_hpc ${SRC_FILES}
        include/profiler_utils.h
        include/old_parallel_kmeans.h
        src/old_parallel_kmeans.cpp)

# --- INCLUDE DIRECTORIES ---
# Dodajemy ścieżki nagłówkowe
target_include_directories(kmeans_hpc PRIVATE include ${MPI_CXX_INCLUDE_PATH})

# --- LINKING ---
# Tutaj była przyczyna błędu. Musimy jawnie dodać "-fopenmp" dla kompilatora GCC/MinGW
target_link_libraries(kmeans_hpc PRIVATE
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
        -fopenmp  # <--- KLUCZOWA POPRAWKA: Flaga dla linkera
)

# Dodatkowe zabezpieczenie dla MinGW (wymuszenie flag linkera)
if(MINGW)
    target_link_options(kmeans_hpc PRIVATE "-fopenmp")
endif()