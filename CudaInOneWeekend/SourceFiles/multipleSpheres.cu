/* Main file to Ray Trace a scene with multiple spheres
 * Still testing to find the best way to allocate CUDA memory
 * for this
*/

#include "header.cuh"

__device__ color rayColor(const ray& r, hittable **world) {
    hit_record rc;

    if ((*world)->hit(r, 0.0f, FLT_MAX, rc)) {
        color pixel_color = 0.5f * (rc.normal + color(1.0f, 1.0f, 1.0f));

        return pixel_color;
    }

    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    color pixel_color = (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);

    return pixel_color;
}

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, hittable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;

    float u = float(i) / float(max_x - 1);
    float v = float(j) / float(max_y - 1);

    ray r(origin, lower_left_corner + u * horizontal + v * vertical);

    color pixel_color = rayColor(r, world);
    getColor(pixel_color);

    fb[pixel_index] = pixel_color;
}

__global__ void allocateWorld(hittable **d_list, hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_world) = new hittable_list(d_list, 2);

        // ground sphere + big sphere
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
    }
}

__global__ void freeWorld(hittable **d_list, hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *(d_list);
        delete *(d_list+1);
        delete *(d_world);
    }
}

int main(void) {
    int nx = 1200;
    int ny = 600;

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

    allocateWorld<<<1, 1>>>(d_list, d_world);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    dim3 blockCount(nx / TX + 1, ny / TY + 1);
    dim3 blockSize(TX, TY);

    render<<<blockCount, blockSize>>> (fb_gpu, nx, ny,
                                      vec3(-2.0f, -1.0f, -1.0f), vec3(4.0f, 0.0f, 0.0f),
                                      vec3(0.0f, 2.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), d_world);

    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    color *fb_cpu = (color*)malloc(num_pixels * sizeof(color));
    cudaStatus = cudaMemcpy(fb_cpu, fb_gpu, num_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    checkReturn(cudaStatus);

    // Output FB as Image
    std::ofstream ppmFile("multiple_spheres.ppm");

    ppmFile << "P3\n" << nx << " " << ny << "\n255\n";

    for (int j = ny - 1; j >= 0; j--) {
        std::cerr << "\rScanlines remaining: " << j << " " << std::flush;
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = static_cast<int>(fb_cpu[pixel_index].e[0]);
            int ig = static_cast<int>(fb_cpu[pixel_index].e[1]);
            int ib = static_cast<int>(fb_cpu[pixel_index].e[2]);
            ppmFile << ir << " " << ig << " " << ib << "\n";
    }
}

    // free world of hittable objects
    freeWorld<<<1, 1>>>(d_list, d_world);
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