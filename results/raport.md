## Wynik profilingu dla wersji sekwencyjnej

==============================
K-Means HPC Project Demo
==============================
Running sequential version...
--- Running Sequential K-Means benchmark ---
Executing 10 runs to average results...
Generating 100000 points in 3 dimensions...
Generation complete!
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 89
Total Wall Time:  1.42086 s
----------------------------------------
1. Initialization:      0.0042428 s (0.298608%)
2. AssignClusters (O(N)): 1.25741 s (88.4961%)
3. UpdateCentroids (O(N)): 0.159212 s (11.2053%)
   ========================================

Run 1/10: 1.4230s (89 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 76
Total Wall Time:  1.2100 s
----------------------------------------
1. Initialization:      0.0043 s (0.3551%)
2. AssignClusters (O(N)): 1.0708 s (88.4925%)
3. UpdateCentroids (O(N)): 0.1349 s (11.1524%)
   ========================================

Run 2/10: 1.2126s (76 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 46
Total Wall Time:  0.7340 s
----------------------------------------
1. Initialization:      0.0045 s (0.6198%)
2. AssignClusters (O(N)): 0.6468 s (88.1227%)
3. UpdateCentroids (O(N)): 0.0826 s (11.2575%)
   ========================================

Run 3/10: 0.7362s (46 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 60
Total Wall Time:  0.9493 s
----------------------------------------
1. Initialization:      0.0043 s (0.4565%)
2. AssignClusters (O(N)): 0.8379 s (88.2625%)
3. UpdateCentroids (O(N)): 0.1071 s (11.2810%)
   ========================================

Run 4/10: 0.9516s (60 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 74
Total Wall Time:  1.1703 s
----------------------------------------
1. Initialization:      0.0046 s (0.3907%)
2. AssignClusters (O(N)): 1.0358 s (88.5007%)
3. UpdateCentroids (O(N)): 0.1300 s (11.1086%)
   ========================================

Run 5/10: 1.1725s (74 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 34
Total Wall Time:  0.5384 s
----------------------------------------
1. Initialization:      0.0044 s (0.8131%)
2. AssignClusters (O(N)): 0.4737 s (87.9900%)
3. UpdateCentroids (O(N)): 0.0603 s (11.1969%)
   ========================================

Run 6/10: 0.5406s (34 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 77
Total Wall Time:  1.2178 s
----------------------------------------
1. Initialization:      0.0043 s (0.3509%)
2. AssignClusters (O(N)): 1.0761 s (88.3667%)
3. UpdateCentroids (O(N)): 0.1374 s (11.2824%)
   ========================================

Run 7/10: 1.2200s (77 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 150
Total Wall Time:  2.3636 s
----------------------------------------
1. Initialization:      0.0047 s (0.1993%)
2. AssignClusters (O(N)): 2.0926 s (88.5332%)
3. UpdateCentroids (O(N)): 0.2663 s (11.2675%)
   ========================================

Run 8/10: 2.3658s (150 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 48
Total Wall Time:  0.7655 s
----------------------------------------
1. Initialization:      0.0046 s (0.5956%)
2. AssignClusters (O(N)): 0.6731 s (87.9219%)
3. UpdateCentroids (O(N)): 0.0879 s (11.4825%)
   ========================================

Run 9/10: 0.7677s (48 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 64
Total Wall Time:  1.0130 s
----------------------------------------
1. Initialization:      0.0045 s (0.4445%)
2. AssignClusters (O(N)): 0.8960 s (88.4516%)
3. UpdateCentroids (O(N)): 0.1125 s (11.1038%)
   ========================================

Run 10/10: 1.0153s (64 iters)

=== Sequential Benchmark Results (10 runs) ===
Total Time (Avg): 1.1405 s
Total Time (Min): 0.5406 s
Total Time (Max): 2.3658 s
Avg Iterations:   71.8000
-------------------------------------------
AVG TIME PER ITERATION: 0.0159 s
-------------------------------------------

## Wynik porównania czasów
==============================
K-Means HPC Project Demo
==============================
--- Running Full Comparison (Seq vs Parallel vs Distributed) ---
Generating dataset (200000 points, 3 dims)...
Generating 200000 points in 3 dimensions...
Generation complete!

1. Sequential K-Means...
   Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 60
Total Wall Time:  1.90286 s
----------------------------------------
1. Initialization:      0.0084387 s (0.443474%)
2. AssignClusters (O(N)): 1.67863 s (88.2161%)
3. UpdateCentroids (O(N)): 0.215792 s (11.3404%)
   ========================================

   Time: 1.90339s, Iters: 60

2. Parallel K-Means (OpenMP)...
   Initializing centroids (Parallel)...
   Time: 0.329977s, Iters: 75

3. Distributed K-Means (MPI)...
   [MPI Rank 0] Initializing centroids...
   Time: 0.880408s, Iters: 111

=== Final Comparison Results ===
Sequential Time: 1.9034 s
Parallel Time:   0.3300 s
Distributed Time:0.8804 s
--------------------------------
Speedup Parallel vs Seq:    5.77x
Speedup Distributed vs Seq: 2.16x
--------------------------------
