#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "HeaderFiles/camera.cuh"
#include "HeaderFiles/hittable_list.cuh"


// Struct to hold the GBuffer texel data
struct GBufferTexel {
    vec3 normal;
    vec3 position;
};

class Render
{   
    public:
        Render(int nx, int ny, cudaGraphicsResource_t cuda_pbo_resource);
        ~Render();

        __host__ void render(float deltaTime, int frame);
        __host__ void denoise();

    private:
        int _nx, _ny;

        camera **_d_cam;
        hittable_list **_d_world;
        curandState *_d_randStatePixels;
        GBufferTexel *_d_gBuffer;
        vec3 *_d_rayTracedImage;
        vec3 *_d_denoisedImage;

        cudaGraphicsResource_t _cuda_pbo_resource; // PBO resource from OpenGL
        uint32_t *_d_output; // Output buffer in device memory
};

#endif
