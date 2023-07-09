/* Header file for our CUDA implement of the Ray Tracer
*/

#ifndef CUDA_HEADER_CUH
#define CUDA_HEADER_CUH

#include <iostream>
#include <fstream>

#include "vec3.cuh"
#include "color.cuh"
#include "ray.cuh"

#define TX 8
#define TY 8

// Adding a method to help us handle errors
inline void checkReturn(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}


#endif