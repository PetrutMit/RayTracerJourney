/* Header file for our CUDA implement of the Ray Tracer
*/

#ifndef CUDA_HEADER_CUH
#define CUDA_HEADER_CUH

#include <fstream>

#include "vec3.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "texture.cuh"
#include "material.cuh"
#include "rotate_y.cuh"
#include "translate.cuh"
#include "constant_medium.cuh"
#include "interval.cuh"
#include "quad.cuh"

#include <curand_kernel.h>

#define TX 8
#define TY 8

//#define FLT_MAX 3.402823466e+38F

// Adding a method to help us handle errors
inline void checkReturn(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}


#endif