Wersja,Czas Całkowity (T),Liczba Iteracji,Czas na 1 iterację,Speedup (vs Seq)
Sequential,12.62 s,127,0.099 s,1.0x (Baza)
OpenMP (Parallel),2.28 s,107,0.021 s,5.53x
MPI (Distributed),3.35 s,118,0.028 s,3.77x

==============================

K-Means HPC Project Demo

==============================

Running sequential version...

--- Running Sequential K-Means benchmark ---

Generating 5000000 points in 3 dimensions...

Generation complete!


Initializing centroids...


========================================

      DETAILED PROFILING REPORT

========================================

Total Iterations: 127

Total Wall Time:  12.617 s

----------------------------------------

1. Initialization:      0.0390501 s (0.309505%)

2. AssignClusters (O(N)): 10.3372 s (81.9306%)

3. UpdateCentroids (O(N)): 2.24075 s (17.7598%)

========================================



Run 1/1 | Wall: 12.6193s | CPU: 12.5938s | Load: 99.8% (127 iters)

Execution finished successfully!





==============================

K-Means HPC Project Demo

==============================

Running parallel OpenMP version...

--- Running Parallel K-Means (Optimized) benchmark ---

Generating 5000000 points in 3 dimensions...

Generation complete!





Initializing centroids (Parallel)...

Run 1/1 | Wall: 2.2790s | CPU: 34.4688s | Load: 1512.5% (107 iters)

Execution finished successfully!



==============================

K-Means HPC Project Demo

==============================

Running distributed MPI version...

--- Running Distributed K-Means (MPI) benchmark ---

Nodes: 8 | Points: 5000000

Generating 5000000 points in 3 dimensions...

Generation complete!

[MPI Rank 0] Initializing centroids...

Logs saved to mpi_log_rank_0.csv (and others)

Run 1/1 | Wall: 3.3469s | Total CPU: 20.8906s | Cluster