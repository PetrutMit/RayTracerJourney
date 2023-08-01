#### Mitrache Mihnea

# Ray-Tracer Journey

## Starting the project
> 1. A good starting point is Peter Shirley's book: [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).
>
> Steps made in this stage can be found in the [InOneWeekend](InOneWeekend) folder.

<hr>

> 2. After understending the concept, it is time to speed up the process using **CUDA**
>
> Steps made in this stage can be found in the [CudaInOneWeekend](CudaInOneWeekend) folder.

<hr>

> 3. After getting a good grasp of the concepts, it is time to start implementing a more complex ray tracer. For this, it is time to look at the next book in series: [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html)
>
> Steps made in this stage can be found in the [TheNextWeek](TheNextWeek) folder.

<hr>

> 4. Since we added lots of functionalities, the render time for Cpp single threaded version is quite big. It is time to speed up the process using **CUDA**.

> Steps made in this stage can be found in the [CudaTheNextWeek](CudaTheNextWeek) folder.

<hr>

> 5. Comparisons between Cpp and CUDA versions and BVH and non-BVH versions were made using  this script [SpeedUpComputer.sh](SpeedUpComputer.sh)
>
> Results after running the script on an i7-10750H CPU @ 2.60GHz and a GeForce RTX 2060 GPU:
```bash
InOneWeekend:
Computing CPU time...
CPU Run 1: 107.349
CPU Run 2: 107.634
CPU Run 3: 103.054
Mean CPU time: 106.012
GPU Run 1: 2.951
GPU Run 2: 1.977
GPU Run 3: 1.975
Mean GPU time: 2.301
InOneWeekend:
CPU: 106.012
GPU: 2.301
Speedup: 46.07214254671881790525
TheNextWeek:
Computing CPU time...
CPU Run 1: 31.265
CPU Run 2: 33.599
CPU Run 3: 32.063
Mean CPU time: 32.309
GPU Run 1: 3.945
GPU Run 2: 2.972
GPU Run 3: 2.964
Mean GPU time: 3
TheNextWeek:
CPU: 32.309
GPU: 3
Speedup: 10.76966666666666666666
BVH impact:
No BVH Run 1: 27.271
No BVH Run 2: 27.451
No BVH Run 3: 27.511
BVH Run 1: 3.033
BVH Run 2: 3.030
BVH Run 3: 3.037
With and without BVH:
No BVH: 27.411
BVH: 3.033
Speedup: 9.03758654797230464886
```