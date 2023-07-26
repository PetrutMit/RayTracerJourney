#include "header.cuh"

#define SPHERE_COUNT 0
#define RECTAGLE_COUNT 8

#define OBJECT_COUNT SPHERE_COUNT + RECTAGLE_COUNT

/* Iterative ray color function
 * Recursive call woulde be:
    return emitted + attenuarion * rec_call
 * So, general recursive formula would be:
 * r(1) = e1 + a1 * r(0)
 * r(2) = e2 + a2 * r(1) => r(2) = e2 + a2 * (e1 + a1 * r(0)) => r(2) = e2 + a2 * e1 + a2 * a1 * r(0)
 * ...
 * r(i) = an * ... * a2 * a1 * r(0) + SIGMA(i = 1, n) e(i) * a(i + 1) * ... * a(n)
* So we use an iterative approach
* Step i would be:
    curAttenuation = curAttenuation * attenuation
    curEmitted = curEmitted + curAttenuation * emitted

*/
__device__ color rayColor(const ray& r, const color& background, hittable **world, curandState *localRandState) {
    ray curRay = r;

    color curAttenuation(1.0f, 1.0f, 1.0f);
    color curEmitted(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 50; i ++) {
        hit_record rec;
        if ((*world)->hit(curRay, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            color attenuation;
            color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            if (rec.mat_ptr->scatter(curRay, rec, attenuation, scattered, localRandState)) {
                curEmitted = curEmitted + curAttenuation * emitted;
                curAttenuation = curAttenuation * attenuation;
                curRay = scattered;
            } else {
                return curAttenuation * emitted + curEmitted;
            }
       } else {
              return curAttenuation * background + curEmitted;
        }
   }
   // exceeded recursion
   return color(0.0f, 0.0f, 0.0f);
}

__global__ void renderInit(int maxX, int maxY, curandState *randStatePixels, curandState *randStateWorld) {
    // Also initialize here the random state for world construction
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, randStateWorld);
    }

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return;

    int pixelIndex = j * maxX + i;

    // Aperently we get better results if we use a different seed for each pixel
    // and same sequence for each thread
    curand_init(1984 + pixelIndex, 0, 0, &randStatePixels[pixelIndex]);
}

__global__ void render(vec3 *fb, int maxX, int maxY, int ns, camera **cam, hittable **world, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return;

    int pixelIndex = j * maxX + i;
    curandState localRandState = randState[pixelIndex];
    color pixelColor(0.0f, 0.0f, 0.0f);
    color background(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < ns; s ++) {
        float u = float(i + curand_uniform(&localRandState)) / float(maxX);
        float v = float(j + curand_uniform(&localRandState)) / float(maxY);

        ray r = (*cam)->get_ray(u, v, &localRandState);
        pixelColor += rayColor(r, background, world, &localRandState);
    }

    getColor(pixelColor, ns);

    fb[pixelIndex] = pixelColor;
}

__global__ void allocateWorld(hittable **d_list, hittable **d_world, camera **d_cam, curandState *randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_world) = new hittable_list(d_list, OBJECT_COUNT);

        lambertian *red = new lambertian(new constant_texture(vec3(0.65f, 0.05f, 0.05f)));
        lambertian *green = new lambertian(new constant_texture(vec3(0.12f, 0.45f, 0.15f)));
        lambertian *white = new lambertian(new constant_texture(vec3(0.73f, 0.73f, 0.73f)));

        diffuse_light *light = new diffuse_light(new constant_texture(vec3(15.0f, 15.0f, 15.0f)));

        *(d_list) = new yz_rect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, green);
        *(d_list + 1) = new yz_rect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, red);
        *(d_list + 2) = new xz_rect(213.0f, 343.0f, 227.0f, 332.0f, 554.0f, light);
        *(d_list + 3) = new xz_rect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, white);
        *(d_list + 4) = new xz_rect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white);
        *(d_list + 5) = new xy_rect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, white);

        hittable *box1 = new box(vec3(0.0f, 0.0f, 0.0f), vec3(165.0f, 330.0f, 165.0f), white);
        box1 = new rotate_y(box1, 15.0f);
        box1 = new translate(box1, vec3(265.0f, 0.0f, 295.0f));

        hittable *box2 = new box(vec3(0.0f, 0.0f, 0.0f), vec3(165.0f, 165.0f, 165.0f), white);
        box2 = new rotate_y(box2, -18.0f);
        box2 = new translate(box2, vec3(130.0f, 0.0f, 65.0f));

        *(d_list + 6) = box1;
        *(d_list + 7) = box2;

        // Camera
        vec3 lookFrom(278.0f, 278.0f, -800.0f);
        vec3 lookAt(278.0f, 278.0f, 0.0f);
        vec3 vUp(0.0f, 1.0f, 0.0f);
        float vfov = 40.0f;

        float distToFocus = 10.0f;
        float aperture = 0.1f;
        float aspect_ratio = 1.0f;

        *d_cam = new camera(lookFrom, lookAt, vUp, vfov, aspect_ratio, aperture, distToFocus, 0.0, 1.0);
    }
}

__global__ void freeWorld(hittable **d_list, hittable **d_world, camera **d_cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < OBJECT_COUNT; i++) {
            delete *(d_list + i);
        }
        delete *(d_world);
        delete *(d_cam);
    }
}

int main(void) {
    int nx = 1200;
    int ny = 1200;
    int ns = 100;

    int num_pixels = nx * ny;

    color *fb_gpu;
    cudaError_t cudaStatus;

    // create device frame buffer
    cudaStatus = cudaMalloc((void**)&fb_gpu, num_pixels * sizeof(color));
    checkReturn(cudaStatus);

    // create random state for each pixel
    curandState *d_randState;
    cudaStatus = cudaMalloc((void**)&d_randState, num_pixels * sizeof(curandState));
    checkReturn(cudaStatus);

    // create random state for world construction
    curandState *d_worldRandState;
    cudaStatus = cudaMalloc((void**)&d_worldRandState, sizeof(curandState));
    checkReturn(cudaStatus);

    // create world of hittable objects
    hittable **d_list;
    cudaStatus = cudaMalloc((void**)&d_list, OBJECT_COUNT * sizeof(hittable*));
    checkReturn(cudaStatus);

    hittable **d_world;
    cudaStatus = cudaMalloc((void**)&d_world, sizeof(hittable*));
    checkReturn(cudaStatus);

    // create camera
    camera **d_cam;
    cudaStatus = cudaMalloc((void**)&d_cam, sizeof(camera*));
    checkReturn(cudaStatus);

    dim3 blockCount(nx + TX - 1 / TX, ny + TY - 1 / TY);
    dim3 blockSize(TX, TY);

    renderInit<<<blockCount, blockSize>>>(nx, ny, d_randState, d_worldRandState);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    allocateWorld<<<1, 1>>>(d_list, d_world, d_cam, d_worldRandState);
    checkReturn(cudaGetLastError());

    // Create events until world is created
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkReturn(cudaDeviceSynchronize());
    checkReturn(cudaEventRecord(start));

    render<<<blockCount, blockSize>>>(fb_gpu, nx, ny, ns, d_cam, d_world, d_randState);

    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    checkReturn(cudaEventRecord(stop));

    float milliseconds = 0;
    checkReturn(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cerr << "Elapsed time: " << milliseconds << " ms\n";

    color *fb_cpu = (color*)malloc(num_pixels * sizeof(color));
    cudaStatus = cudaMemcpy(fb_cpu, fb_gpu, num_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    checkReturn(cudaStatus);

    // Output FB as Image
    std::ofstream ppmFile("cornell_box_with_transformed_boxes.ppm");

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

    cudaStatus = cudaFree(d_cam);
    checkReturn(cudaStatus);

    cudaStatus = cudaFree(d_randState);
    checkReturn(cudaStatus);

    cudaStatus = cudaFree(d_worldRandState);
    checkReturn(cudaStatus);

    free(fb_cpu);
    std::cerr << "\nDone.\n";
}