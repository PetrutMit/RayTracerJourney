#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include "vec3.cuh"

class Render
{   
    public:
        Render(GLuint nx, GLuint ny, cudaGraphicsResource_t cuda_pbo_resource);

        // Kernel functions
        __global__ void renderInit(int maxX, int maxY, curandState *randStatePixels, curandState *randStateWorld);
        __global__ void allocateWorld(hittable **d_list, hittable **d_world, camera **d_cam, curandState *randState);
        __global__ void raytrace(vec3 *fb, int maxX, int maxY, int ns, camera **cam, hittable **world, curandState *randState);
        __global__ void freeWorld(hittable **d_list, hittable **d_world, camera **d_cam);
        __device__ color rayColor(ray &r, color &background, hittable **world, curandState *localRandState);

        // Host functions
        __host__ void render();
        __host__ void free();

    private:
        GLuint _nx, _ny;
        GLuint _ns;

        camera **_d_cam;
        hittable **_d_world;
        curandState *_d_randStatePixels;



        cudaGraphicsResource_t _cuda_pbo_resource; // PBO resource from OpenGL
        // By using cudaGraphicsResourceGetMappedPointer we can get the pointer to use in the kernel
};