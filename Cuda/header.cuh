/* Header file for our CUDA implement of the Ray Tracer
*/

#ifndef CUDA_HEADER_CUH
#define CUDA_HEADER_CUH

#include <iostream>
#include <fstream>

#include "vec3.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"

#include <curand_kernel.h>

#define TX 8
#define TY 8

#define FLT_MAX 3.402823466e+38F

// Adding a method to help us handle errors
inline void checkReturn(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Random auxilliary functions
__device__ vec3 randomVector(curandState *localRandState) {
    float a = curand_uniform(localRandState);
    float b = curand_uniform(localRandState);
    float c = curand_uniform(localRandState);
    return unit_vector(vec3(a,b,c));
}

__device__ vec3 randomInUnitSphere(curandState *localRandState) {
    vec3 p;
    do {
        p = 2.0f * randomVector(localRandState) - vec3(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

#endif