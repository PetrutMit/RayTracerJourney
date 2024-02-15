### Ray Tracing Journey 

**Student** : Mihnea Mitrache

**Coordinator** : Victor Asavei

<hr>

## Short Project Description
> The goal of this project was to implement the **Ray Tracing** rendering algorithm
in various ways, using different parallelization techniques. All in order to reach
the best performance possible, to get close to real-time rendering.

<hr>

## Steps taken
> An excellent starting point to get a hands on experience with a Ray Tracer
was Peter Shirley's book ["Ray Tracing in One Weekend"](https://raytracing.github.io/books/RayTracingInOneWeekend.html).
I made a thorough analysis of various CPU and GPU implements of that Ray Tracer,
which can be found [here](/Analysis/).

> This proved that CUDA solution has lots of potential. A step by step implementation
of multiple RT scenes in CUDA can be found [here](/CudaInOneWeekend/)

> I also explored further improvements and features that can be added to the Ray Tracer.
As a guideline, I used the book ["Ray Tracing: The Next Week"](https://raytracing.github.io/books/RayTracingTheNextWeek.html).
The most significant improvement was the implementation of a BVH (Bounding Volume Hierarchy) for the Ray Tracer. This was quite a challenge in CUDA, but the results were very satisfying.

