# Script which compares Cpp CPU implements and CUDA GPU implements
# Sources were modified to remove any output to the console so that
# the time is spent mostly on the computation
#! /bin/bash

# Setting time format
TIMEFORMAT='%3R'

# InOneWeekend
echo "InOneWeekend:"

# CPU
cd InOneWeekend
cd Chapters12-13
make clean 1>/dev/null 2>&1
make final_scene 1>/dev/null 2>&1

tCPU=0.0
echo "Computing CPU time..."
for i in {1..3}
do
    t=$({ time ./final_scene 2>&1 >/dev/null; } 2>&1)
    echo "CPU Run $i: $t"
    tCPU=$(echo "$tCPU + $t" | bc)
done

tCPU=$(echo "scale=3; $tCPU / 3" | bc)
echo "Mean CPU time: $tCPU"

# GPU
cd ../..
cd CudaInOneWeekend
cd SourceFiles
make clean 1>/dev/null 2>&1
make finalScene 1>/dev/null 2>&1

tGPU=0.0
for i in {1..3}
do
    t=$({ time ./finalScene 2>&1 >/dev/null; } 2>&1)
    echo "GPU Run $i: $t"
    tGPU=$(echo "$tGPU + $t" | bc)
done

tGPU=$(echo "scale=3; $tGPU / 3" | bc)
echo "Mean GPU time: $tGPU"

# Print mean times and speedup
echo "InOneWeekend:"
echo "CPU: $tCPU"
echo "GPU: $tGPU"
echo "Speedup: $(echo "$tCPU / $tGPU" | bc -l)"

# TheNextWeek
echo "TheNextWeek:"

cd ../..
cd TheNextWeek
cd Chapter10
make clean 1>/dev/null 2>&1
make final_scene 1>/dev/null 2>&1

tCPU=0.0
echo "Computing CPU time..."
for i in {1..3}
do
    t=$({ time ./final_scene 2>&1 >/dev/null; } 2>&1)
    echo "CPU Run $i: $t"
    tCPU=$(echo "$tCPU + $t" | bc)
done


tCPU=$(echo "scale=3; $tCPU / 3" | bc)
echo "Mean CPU time: $tCPU"

# GPU
cd ../..
cd CudaTheNextWeek
cd SourceFiles
make clean 1>/dev/null 2>&1
make finalScene 1>/dev/null 2>&1

tGPU=0.0
for i in {1..3}
do
    t=$({ time ./finalScene 1 2>&1 >/dev/null; } 2>&1)
    echo "GPU Run $i: $t"
    tGPU=$(echo "$tGPU + $t" | bc)
done

tGPU=$(echo "$tGPU / 3" | bc)
echo "Mean GPU time: $tGPU"

# Print mean times and speedup
echo "TheNextWeek:"
echo "CPU: $tCPU"
echo "GPU: $tGPU"
echo "Speedup: $(echo "$tCPU / $tGPU" | bc -l)"

# For CUDA Implement, see how much of a difference the bvh makes
# This one uses the internal events for better accuracy
# Code was modified to print only the render time to stdout
echo "BVH impact:"

tNO_BVH=0.0

for i in {1..3}
do
    t=$({ time ./finalScene 0 1>/dev/null 2>&1; } 2>&1)
    echo "No BVH Run $i: $t"
    tNO_BVH=$(echo "$tNO_BVH + $t" | bc)
done

tNO_BVH=$(echo "scale=3; $tNO_BVH / 3" | bc)

tBVH=0.0

for i in {1..3}
do
    t=$({ time ./finalScene 1 1>/dev/null 2>&1; } 2>&1)
    echo "BVH Run $i: $t"
    tBVH=$(echo "$tBVH + $t" | bc)
done

tBVH=$(echo "scale=3; $tBVH / 3" | bc)

echo "With and without BVH:"
echo "No BVH: $tNO_BVH"
echo "BVH: $tBVH"
echo "Speedup: $(echo "$tNO_BVH / $tBVH" | bc -l)"

