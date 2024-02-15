#! /bin/bash

# Mihnea Mitrache

# Script which compares times for:
#   1. CPU, single threaded CPP version
#   2. CPU, multi threaded OpenMP version
#   3. CPU, multi process MPI version
#   4. CPU, multi thread Pthread version
#   5. CPU, hybrid MPI + OpenMP version
#   6. GPU, CUDA version

# Each implement computes the same 900 * 600 image
# Using 10 samples per pixel

# If the flag --time is passed:
# The script runs each implementation RUNS times
# and computes the mean time for each implementation with a fixed
# number of threads for the parallel versions 
# Example: ./speedUp.sh --time

# If the flag --speedup is passed:
# The script computes the speedup for the parallel versions 
# for various number of threads [2, 4, 8, 12, 16]

# Number of runs
RUNS=3

# Setting time format
TIMEFORMAT='%3R'

# If the flag --time is passed
if [ "$1" = "--time" ]
then
    echo "Single threaded CPP:"

    # 1. Single threaded CPP
    cd RT_Serial_CPP
    make clean 1>/dev/null 2>&1
    make final_scene 1>/dev/null 2>&1

    tSerialCPU=0.0
    echo "Computing CPU serial time..."
    for i in $(seq 1 $RUNS)
    do
        t=$({ time ./final_scene 2>&1 >/dev/null; } 2>&1)
        echo "CPU Run $i: $t"
        tSerialCPU=$(echo "$tSerialCPU + $t" | bc)
    done

    tSerialCPU=$(echo "scale=$RUNS; $tSerialCPU / $RUNS" | bc)
    echo "Mean CPU serial time in seconds: $tSerialCPU"

    # 2. Multi threaded using OpenMP
    cd ..
    cd RT_Parallel_OPENMP
    make clean 1>/dev/null 2>&1
    make final_scene 1>/dev/null 2>&1

    tParallelOMP=0.0
    echo "Computing CPU parallel OpenMP time..."
    for i in $(seq 1 $RUNS)
    do
        t=$({ time ./final_scene 4 2>&1 >/dev/null; } 2>&1)
        echo "OpenMP Run $i: $t"
        tParallelOMP=$(echo "$tParallelOMP + $t" | bc)
    done

    tParallelOMP=$(echo "scale=$RUNS; $tParallelOMP / $RUNS" | bc)
    echo "Mean CPU parallel OpenMP time in seconds: $tParallelOMP"

    # 3. Multi process using MPI
    cd ..
    cd RT_Parallel_MPI
    make clean 1>/dev/null 2>&1
    make final_scene_build 1>/dev/null 2>&1

    tParallelMPI=0.0
    echo "Computing CPU parallel MPI time..."
    for i in $(seq 1 $RUNS)
    do
        t=$({ time mpirun -np 4 ./final_scene 2>&1 >/dev/null; } 2>&1)
        echo "MPI Run $i: $t"
        tParallelMPI=$(echo "$tParallelMPI + $t" | bc)
    done

    tParallelMPI=$(echo "scale=$RUNS; $tParallelMPI / $RUNS" | bc)
    echo "Mean CPU parallel MPI time in seconds: $tParallelMPI"

    # 4. Multi thread using PTHREAD
    cd ..
    cd RT_Parallel_PTHREADS
    make clean 1>/dev/null 2>&1
    make final_scene 1>/dev/null 2>&1

    tParallelPTHREAD=0.0
    echo "Computing CPU parallel PTHREAD time..."
    for i in $(seq 1 $RUNS)
    do
        t=$({ time ./final_scene 4 2>&1 >/dev/null; } 2>&1)
        echo "PTHREAD Run $i: $t"
        tParallelPTHREAD=$(echo "$tParallelPTHREAD + $t" | bc)
    done

    tParallelPTHREAD=$(echo "scale=$RUNS; $tParallelPTHREAD / $RUNS" | bc)
    echo "Mean CPU parallel PTHREAD time in seconds: $tParallelPTHREAD"

   # 5. Hybrid MPI + OpenMP
    cd ..
    cd RT_Parallel_Hybrid_MPI_OPENMP
    make clean 1>/dev/null 2>&1
    make final_scene_build 1>/dev/null 2>&1

    tParallelHybrid=0.0
    echo "Computing CPU parallel MPI + OPENMP time..."
    for i in $(seq 1 $RUNS)
    do
        t=$({ time mpirun -np 4 ./final_scene 4 2>&1 >/dev/null; } 2>&1)
        echo "MPI + OPENMP Run $i: $t"
        tParallelHybrid=$(echo "$tParallelHybrid + $t" | bc)
    done

    tParallelHybrid=$(echo "scale=$RUNS; $tParallelHybrid / $RUNS" | bc)
    echo "Mean CPU parallel PTHREAD time in seconds: $tParallelHybrid"

    # 6. GPU, CUDA version
    cd ..
    cd RT_Parallel_CUDA
    make clean 1>/dev/null 2>&1
    make final_scene 1>/dev/null 2>&1

    tParallelCUDA=0.0
    echo "Computing GPU parallel CUDA time..."
    for i in $(seq 1 $RUNS)
    do
        t=$({ time ./final_scene 2>&1 >/dev/null; } 2>&1)
        echo "GPU-CUDA Run $i: $t"
        tParallelCUDA=$(echo "$tParallelCUDA + $t" | bc)
    done

    tParallelCUDA=$(echo "scale=$RUNS; $tParallelCUDA / $RUNS" | bc)
    echo "Mean GPU parallel CUDA time in seconds: $tParallelCUDA"
fi

# If the flag --speedup is passed
if [ "$1" = "--speedup" ]
then
    # 0. Single threaded CPP - for computing speedup
    cd RT_Serial_CPP
    make clean 1>/dev/null 2>&1
    make final_scene 1>/dev/null 2>&1
    tSerialCPU=0.0
    echo "Computing serial time to calculate speedup..."
    tSerialCPU=$({ time ./final_scene 2>&1 >/dev/null; } 2>&1)
    echo "Serial time in seconds: $tSerialCPU"

for N_THREADS in 2 4 8 12 16
do
    echo "Computing speedup for $N_THREADS threads..."
    echo "-----------------------------------------"

    # 1. Multi threaded using OpenMP Speedup
    cd ..
    cd RT_Parallel_OPENMP
    make clean 1>/dev/null 2>&1
    make final_scene 1>/dev/null 2>&1
    tParallelOMP=0.0
    tParallelOMP=$({ time ./final_scene $N_THREADS 2>&1 >/dev/null; } 2>&1)
    echo "OpenMP time in seconds with $N_THREADS threads: $tParallelOMP"
    echo "Speedup for OpenMP: $(echo "scale=2; $tSerialCPU / $tParallelOMP" | bc)"

    # 2. Multi process using MPI Speedup
    cd ..
    cd RT_Parallel_MPI
    make clean 1>/dev/null 2>&1
    make final_scene_build 1>/dev/null 2>&1
    tParallelMPI=0.0
    tParallelMPI=$({ time mpirun -np $N_THREADS --oversubscribe ./final_scene 2>&1 >/dev/null; } 2>&1)
    echo "MPI time in seconds with $N_THREADS threads: $tParallelMPI"
    speedUP=$(echo "scale=2; $tSerialCPU / $tParallelMPI" | bc)
    echo "Speedup for MPI: $speedUP"

    # 3. Multi thread using PTHREAD Speedup
    cd ..
    cd RT_Parallel_PTHREADS
    make clean 1>/dev/null 2>&1
    make final_scene 1>/dev/null 2>&1
    tParallelPTHREAD=0.0
    tParallelPTHREAD=$({ time ./final_scene $N_THREADS 2>&1 >/dev/null; } 2>&1)
    echo "PTHREAD time in seconds with $N_THREADS threads: $tParallelPTHREAD"
    speedUP=$(echo "scale=2; $tSerialCPU / $tParallelPTHREAD" | bc)
    echo "Speedup for PTHREAD: $speedUP"

    # 4. Hybrid MPI + OpenMP Speedup
    cd ..
    cd RT_Parallel_Hybrid_MPI_OPENMP
    make clean 1>/dev/null 2>&1
    make final_scene_build 1>/dev/null 2>&1
    tParallelHybrid=0.0
    tParallelHybrid=$({ time mpirun -np $N_THREADS --oversubscribe ./final_scene $N_THREADS 2>&1 >/dev/null; } 2>&1)
    echo "MPI + OPENMP time in seconds with $N_THREADS threads: $tParallelHybrid"
    speedUP=$(echo "scale=2; $tSerialCPU / $tParallelHybrid" | bc)
    echo "Speedup for MPI + OPENMP: $speedUP"

    echo "-----------------------------------------"
done

fi