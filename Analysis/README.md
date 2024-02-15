## Structure
* Solutions for our problem:
    * [Serial solution](RT_Serial_CPP)
    * [OpenMP solution](RT_Parallel_OPENMP)
    * [MPI solution](RT_Parallel_MPI)
    * [Pthreads solution](RT_Parallel_PTHREADS)
    * [CUDA solution](RT_Parallel_CUDA)
    * [Hybrid OPENMP + MPI solution](RT_Parallel_OPENMP_MPI)
* Profiling analysis for each of the above solutions:
    * [Serial profiling](Profiling/Serial/)
    * [OpenMP profiling](Profiling/OPENMP/)
    * [MPI profiling](Profiling/MPI/)
    * [Pthreads profiling](Profiling/PTHREADS/)
    * [CUDA profiling](Profiling/CUDA/)
    * [Hybrid OPENMP + MPI profiling](Profiling/Hybrid/)
* Performance analysis:
    * Bash script for computing mean time and speedup for solutions 
        * [performance.sh](Performance/performance.sh)
    * MATLAB script for graph analysis
        * [graphs.m](Performance/graphs.m)
    * Computed graphs
        * [Graphs](Performance/Graphs)
            * [10_SPP](Performance/Graphs/10spp)
            * [2_SPP](Performance/Graphs/2spp)
<hr>

## Progress log
1. Week 1
    * [x] Serial solution
    * [x] Profiling for serial solution
    * [x] Identify parallelization opportunities
    * [x] Project initial readme 
2. Week 2
    * [x] CUDA solution
    * [x] Profiling for CUDA solution
3. Week 3
    * [x] OpenMP solution
    * [x] Profiling for OpenMP solution
    * [x] MPI solution 
    * [x] Profiling for MPI solution
4. Week 4
    * [x] Pthreads solution
    * [x] Profiling for Pthreads solution
    * [x] Create script for computing mean time
5. Week 5
    * [x] Reorganize project structure
    * [x] Project wiki page
6. Week 6
    * [x] Hybrid OPENMP + MPI solution
    * [x] Profiling for hybrid OPENMP + MPI solution 
    * [x] Prepare code for graph analysis   
7. Week 7
    * [x] Modify script to also compute speedup with various number of threads/processes
    * [x] Added one more test scenario
8. Week 8  
    * [x] Added a MATLAB script for graph analysis
<hr>

## Results

> Results were evaluated based on the provided script. Each solution was run 3 times to get
mean time and speedup. The results are presented below.

> The results were computed on a machine with the following specs:
   * CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
   * GPU: NVIDIA GeForce RTX 2060
```bash
Single threaded CPP:
Computing CPU serial time...
Mean CPU serial time in seconds: 273.960

Computing CPU parallel OpenMP time...
Mean CPU parallel OpenMP time in seconds: 100.722
Speedup: 2.72

Computing CPU parallel MPI time...
Mean CPU parallel MPI time in seconds: 115.253
Speedup: 2.38

Computing CPU parallel PTHREAD time...
Mean CPU parallel PTHREAD time in seconds: 86.083
Speedup: 3.18

Computing CPU parallel MPI + OPENMP time...
Mean CPU parallel MPI + OPENMP time in seconds: 109.307
Speedup: 2.51

Computing GPU parallel CUDA time...
Mean GPU parallel CUDA time in seconds: 7.548
Speedup: 36.31
```

### Graph Analysis
> Using a MATLAB script, various graphs were plotted. The graphs use the data from the above section.
Those were created and saved in .png format automatically from the script.
1. 10 SPP
 * Execution time
![1](/Analysis/Performance/Graphs/10spp/1.png)
 * Comparison between serial, best CPU parallel and GPU powered solutions

The line chart shows how mean execution time varies with the number of threads/processes for the CPU parallel solutions. It is clear that each solution scales well.
The best solution is the PTHREADS one. Slighly worse is the OPENMP one. This is because OPEN MP is more simple to use and
this simplicity comes with the cost of more overhead.
The MPI and Hybrid solutions follow a similar trend.
<hr>

![2](/Analysis/Performance/Graphs/10spp/2.png)

The bar chart shows a comparison between serial, best CPU parallel and GPU powered solutions. It is clear that the CUDA solution is the absolute winner. No CPU based algorithm can compete with it. The best CPU parallel solution is the PTHREADS one. Comparing it with the serial one, it is far better.
<hr>

![3](/Analysis/Performance/Graphs/10spp/3.png)
![4](/Analysis/Performance/Graphs/10spp/4.png)

The above charts show how speedup varies with the number of threads/processes for the CPU parallel solutions. Each solutions scales well. The rate of scaling decreases because of Amdahl's law.
<hr>

![5](/Analysis/Performance/Graphs/10spp/5.png)

The last chart shows the gap, in speedup, between best CPU parallel solution and GPU powered one. CUDA speedup is 10 times greater than PTHREADS.

1. 2SPP
![1](/Analysis/Performance/Graphs/2spp/1.png)
![2](/Analysis/Performance/Graphs/2spp/2.png)
![3](/Analysis/Performance/Graphs/2spp/3.png)
![4](/Analysis/Performance/Graphs/2spp/4.png)
![5](/Analysis/Performance/Graphs/2spp/5.png)

> In contrast with the higher quality image, the gap between CPU parallel solutions is smaller. This is because we have less parallel chunks for the same boilerplate. 