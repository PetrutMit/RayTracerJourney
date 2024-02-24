#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "HeaderFiles/camera.cuh"
#include "HeaderFiles/hittable.cuh"

class Render
{   
    public:
        Render(int nx, int ny, cudaGraphicsResource_t cuda_pbo_resource);

        // Host functions
        __host__ void render();
        __host__ void free();

    private:
        int _nx, _ny;

        camera **_d_cam;
        hittable **_d_world;
        curandState *_d_randStatePixels;

        cudaGraphicsResource_t _cuda_pbo_resource; // PBO resource from OpenGL
        // By using cudaGraphicsResourceGetMappedPointer we can get the pointer to use in the kernel
};

#endif
