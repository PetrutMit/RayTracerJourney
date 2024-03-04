## Ray Tracing Journey 

**Student** : Mihnea Mitrache

**Coordinator** : Victor Asavei

<hr>

### Short Project Description
The `goal` of this project is to `exhaustively explore` the `Ray Tracing` algorithm. Different `implementations,
optimizations and features` are considered. All in order to reach the best performance possible, to get close to `real-time rendering`.
<hr>

## Steps taken
1. **Ground analysis**

An excellent starting point to get a hands on experience with a Ray Tracer was Peter Shirley's book ["Ray Tracing in One Weekend"](https://raytracing.github.io/books/RayTracingInOneWeekend.html).
I made a thorough analysis of various CPU and GPU implements of that Ray Tracer, which can be found [here](/Analysis/).
<hr>

2. **In depth CUDA implementation**

This proved that CUDA solution has lots of potential. A step by step implementation of multiple RT scenes in CUDA can be found [here](/CudaInOneWeekend/)
<hr>

3. **Further improvements**

I also explored further improvements and features that can be added to the Ray Tracer.
As a guideline, I used the book ["Ray Tracing: The Next Week"](https://raytracing.github.io/books/RayTracingTheNextWeek.html).
A step by step feature addition can be found [here](/CudaTheNextWeek/).

The `most significant improvement` was the implementation of a `BVH (Bounding Volume Hierarchy)` for the Ray Tracer. This was quite a challenge in CUDA, but the results were very satisfying. 
In this point, if only kernel execution times are considered, the Ray Tracer is `close to real-time rendering`.
<hr>

4. **Interactivity**

Last optimizations brought the hope of real-time rendering. In this part of the project, `dynamic scenes` are explored. The steps taken can be found [here](/Interactive/).


