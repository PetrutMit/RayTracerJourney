/* Main file to Ray Trace a scene with multiple spheres
 * Still testing to find the best way to allocate CUDA memory
 * for this
*/

#include "header.cuh"

__device__ color rayColor(const ray& r, hittable **world, curandState *localRandState) {
   ray curRay = r;

   float curAttenuation = 1.0f;

   for (int i = 0; i < 50; i ++) {
        hit_record rec;
        if ((*world)->hit(curRay, 0.001f, FLT_MAX, rec)) {
            vec3 target = rec.p + rec.normal + randomInHemiSphere(rec.normal, localRandState);
            curAttenuation *= 0.5f;
            curRay = ray(rec.p, target - rec.p);
        } else {
            vec3 unitDirection = unit_vector(curRay.direction());
            float t = 0.5f * (unitDirection.y() + 1.0f);
            color c1(1.0f, 1.0f, 1.0f);
            color c2(0.5f, 0.7f, 1.0f);
            return curAttenuation * ((1.0f - t) * c1 + t * c2);
        }
   }
   // exceeded recursion
   return color(0.0f, 0.0f, 0.0f);
}

__global__ void renderInit(int maxX, int maxY, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return;

    int pixelIndex = j * maxX + i;

    // Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixelIndex, 0, &randState[pixelIndex]);
}

__global__ void render(vec3 *fb, int maxX, int maxY, int ns, camera **cam, hittable **world, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return;

    int pixelIndex = j * maxX + i;
    curandState localRandState = randState[pixelIndex];
    color pixelColor(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < ns; s ++) {
        float u = float(i + curand_uniform(&localRandState)) / float(maxX);
        float v = float(j + curand_uniform(&localRandState)) / float(maxY);

        ray r = (*cam)->get_ray(u, v);
        pixelColor += rayColor(r, world, &localRandState);
    }

    getColor(pixelColor, ns);

    fb[pixelIndex] = pixelColor;
}

__global__ void allocateWorld(hittable **d_list, hittable **d_world, camera **d_cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_world) = new hittable_list(d_list, 2);
        *(d_cam) = new camera();

        // ground sphere + big sphere
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
    }
}

__global__ void freeWorld(hittable **d_list, hittable **d_world, camera **d_cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *(d_list);
        delete *(d_list+1);
        delete *(d_world);
        delete *(d_cam);
    }
}

int main(void) {
    int nx = 1200;
    int ny = 600;
    int ns = 100;

    int num_pixels = nx * ny;

    color *fb_gpu;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&fb_gpu, num_pixels * sizeof(color));
    checkReturn(cudaStatus);

    // create world of hittable objects
    hittable **d_list;
    cudaStatus = cudaMalloc((void**)&d_list, 2 * sizeof(hittable*));
    checkReturn(cudaStatus);

    hittable **d_world;
    cudaStatus = cudaMalloc((void**)&d_world, sizeof(hittable*));
    checkReturn(cudaStatus);

    // create camera
    camera **d_cam;
    cudaStatus = cudaMalloc((void**)&d_cam, sizeof(camera*));

    allocateWorld<<<1, 1>>>(d_list, d_world, d_cam);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    // create random state
    curandState *d_randState;
    cudaStatus = cudaMalloc((void**)&d_randState, num_pixels * sizeof(curandState));
    checkReturn(cudaStatus);


    dim3 blockCount(nx / TX + 1, ny / TY + 1);
    dim3 blockSize(TX, TY);

    renderInit<<<blockCount, blockSize>>>(nx, ny, d_randState);

    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    render<<<blockCount, blockSize>>>(fb_gpu, nx, ny, ns, d_cam, d_world, d_randState);

    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    color *fb_cpu = (color*)malloc(num_pixels * sizeof(color));
    cudaStatus = cudaMemcpy(fb_cpu, fb_gpu, num_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    checkReturn(cudaStatus);

    // Output FB as Image
    std::ofstream ppmFile("diffuse_spheres.ppm");

    ppmFile << "P3\n" << nx << " " << ny << "\n255\n";

    for (int j = ny - 1; j >= 0; j--) {
        std::cerr << "\rScanlines remaining: " << j << " " << std::flush;
        for (int i = 0; i < nx; i++) {
            size_t pixelIndex = j * nx + i;
            int ir = static_cast<int>(fb_cpu[pixelIndex].e[0]);
            int ig = static_cast<int>(fb_cpu[pixelIndex].e[1]);
            int ib = static_cast<int>(fb_cpu[pixelIndex].e[2]);
            ppmFile << ir << " " << ig << " " << ib << "\n";
    }
}

    // free world of hittable objects
    freeWorld<<<1, 1>>>(d_list, d_world, d_cam);
    checkReturn(cudaGetLastError());

    cudaStatus = cudaFree(d_list);
    checkReturn(cudaStatus);

    cudaStatus = cudaFree(d_world);
    checkReturn(cudaStatus);

    cudaStatus = cudaFree(fb_gpu);
    checkReturn(cudaStatus);

    free(fb_cpu);
    std::cerr << "\nDone.\n";
}