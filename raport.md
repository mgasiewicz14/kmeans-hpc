==============================
K-Means HPC Project Demo
==============================
Running sequential version...
--- Running Sequential K-Means benchmark ---
Executing 10 runs to average results...
Generating 200000 points in 3 dimensions...
Generation complete!
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 100
Total Wall Time:  11.916 s
----------------------------------------
1. Initialization:      0.0359231 s (0.30147%)
2. AssignClusters (O(N)): 9.81884 s (82.4005%)
3. UpdateCentroids (O(N)): 2.06123 s (17.298%)
   ========================================

Run 1/10: 11.9183s (100 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 108
Total Wall Time:  12.9058 s
----------------------------------------
1. Initialization:      0.0362 s (0.2801%)
2. AssignClusters (O(N)): 10.6276 s (82.3477%)
3. UpdateCentroids (O(N)): 2.2420 s (17.3722%)
   ========================================

Run 2/10: 12.9082s (108 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 121
Total Wall Time:  14.4509 s
----------------------------------------
1. Initialization:      0.0363 s (0.2512%)
2. AssignClusters (O(N)): 11.9051 s (82.3831%)
3. UpdateCentroids (O(N)): 2.5095 s (17.3657%)
   ========================================

Run 3/10: 14.4533s (121 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 68
Total Wall Time:  8.1261 s
----------------------------------------
1. Initialization:      0.0365 s (0.4497%)
2. AssignClusters (O(N)): 6.6815 s (82.2225%)
3. UpdateCentroids (O(N)): 1.4081 s (17.3278%)
   ========================================

Run 4/10: 8.1283s (68 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 58
Total Wall Time:  6.9287 s
----------------------------------------
1. Initialization:      0.0364 s (0.5253%)
2. AssignClusters (O(N)): 5.6956 s (82.2039%)
3. UpdateCentroids (O(N)): 1.1966 s (17.2707%)
   ========================================

Run 5/10: 6.9309s (58 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 123
Total Wall Time:  14.6787 s
----------------------------------------
1. Initialization:      0.0362 s (0.2469%)
2. AssignClusters (O(N)): 12.1086 s (82.4911%)
3. UpdateCentroids (O(N)): 2.5338 s (17.2620%)
   ========================================

Run 6/10: 14.6810s (123 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 100
Total Wall Time:  11.9622 s
----------------------------------------
1. Initialization:      0.0363 s (0.3038%)
2. AssignClusters (O(N)): 9.8557 s (82.3905%)
3. UpdateCentroids (O(N)): 2.0701 s (17.3056%)
   ========================================

Run 7/10: 11.9645s (100 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 90
Total Wall Time:  10.7417 s
----------------------------------------
1. Initialization:      0.0366 s (0.3406%)
2. AssignClusters (O(N)): 8.8472 s (82.3631%)
3. UpdateCentroids (O(N)): 1.8579 s (17.2963%)
   ========================================

Run 8/10: 10.7443s (90 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 80
Total Wall Time:  9.5796 s
----------------------------------------
1. Initialization:      0.0362 s (0.3782%)
2. AssignClusters (O(N)): 7.8821 s (82.2801%)
3. UpdateCentroids (O(N)): 1.6613 s (17.3417%)
   ========================================

Run 9/10: 9.5819s (80 iters)
Initializing centroids...

========================================
DETAILED PROFILING REPORT
========================================
Total Iterations: 65
Total Wall Time:  7.7921 s
----------------------------------------
1. Initialization:      0.0368 s (0.4729%)
2. AssignClusters (O(N)): 6.4060 s (82.2112%)
3. UpdateCentroids (O(N)): 1.3493 s (17.3159%)
   ========================================

Run 10/10: 7.7943s (65 iters)

=== Sequential Benchmark Results (10 runs) ===
Total Time (Avg): 10.9105 s
Total Time (Min): 6.9309 s
Total Time (Max): 14.6810 s
Avg Iterations:   91.3000
-------------------------------------------
AVG TIME PER ITERATION: 0.1195 s
-------------------------------------------