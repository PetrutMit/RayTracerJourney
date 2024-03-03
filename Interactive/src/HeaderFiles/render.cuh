#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "HeaderFiles/camera.cuh"
#include "HeaderFiles/hittable_list.cuh"

class Render
{   
    public:
        Render(int nx, int ny, cudaGraphicsResource_t cuda_pbo_resource);
        ~Render();

        __host__ void render(float deltaTime);

    private:
        int _nx, _ny;

        camera **_d_cam;
        hittable_list **_d_world;
        curandState *_d_randStatePixels;

        cudaGraphicsResource_t _cuda_pbo_resource; // PBO resource from OpenGL
        uint32_t *_d_output; // Output buffer in device memory
};

#endif
