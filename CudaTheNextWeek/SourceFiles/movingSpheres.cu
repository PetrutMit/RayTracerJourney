#include "header.cuh"

#define SPHERE_COUNT 489

__device__ color rayColor(const ray& r, hittable **world, curandState *localRandState) {
   ray curRay = r;

    color curAttenuation(1.0f, 1.0f, 1.0f);

   for (int i = 0; i < 50; i ++) {
        hit_record rec;
        if ((*world)->hit(curRay, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(curRay, rec, attenuation, scattered, localRandState)) {
                curAttenuation = curAttenuation * attenuation;
                curRay = scattered;
            } else {
                return color(0.0f, 0.0f, 0.0f);
            }
       } else {
            vec3 unitDirection = unit_vector(curRay.direction());
            float t = 0.5f * (unitDirection.y() + 1.0f);
            color c1(1.0f, 1.0f, 1.0f);
            color c2(0.9f, 0.9f, 1.0f);
            return curAttenuation * ((1.0f - t) * c1 + t * c2);
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
    for (int s = 0; s < ns; s ++) {
        float u = float(i + curand_uniform(&localRandState)) / float(maxX);
        float v = float(j + curand_uniform(&localRandState)) / float(maxY);

        ray r = (*cam)->get_ray(u, v, &localRandState);
        pixelColor += rayColor(r, world, &localRandState);
    }

    getColor(pixelColor, ns);

    fb[pixelIndex] = pixelColor;
}

__global__ void allocateWorld(hittable **d_list, hittable **d_world, camera **d_cam, curandState *d_randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_world) = new hittable_list(d_list,SPHERE_COUNT);

        curandState localRandState = *d_randState;

        // Spheres
        material *groundMat = new lambertian(vec3(0.5f, 0.5f, 0.5f), false);
        *(d_list) = new moving_sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, groundMat);
        int cnt = 1;
        for (int a = -11; a < 11; a ++) {
            for (int b = -11; b < 11; b ++) {
                float chooseMat = curand_uniform(&localRandState);
                vec3 center(a + curand_uniform(&localRandState), 0.2f, b + curand_uniform(&localRandState));
                material *sphereMat;
                    if (chooseMat < 0.8f) {
                        // diffuse
                        vec3 albedo = vec3(curand_uniform(&localRandState) * curand_uniform(&localRandState),
                                           curand_uniform(&localRandState) * curand_uniform(&localRandState),
                                           curand_uniform(&localRandState) * curand_uniform(&localRandState));
                        sphereMat = new lambertian(albedo, false);
                        vec3 center2 = center + vec3(0.0f, curand_uniform(&localRandState) * 0.5f, 0.0f);
                        moving_sphere *m = new moving_sphere(center, 0.2f, sphereMat, 0.0f, 1.0f, center2);
                        *(d_list + cnt++) = m;
                        //*(d_list + cnt++) = new sphere(center, 0.2f, sphereMat);
                    } else if (chooseMat < 0.95f) {
                        // metal
                        vec3 albedo = vec3(0.5f * (1.0f + curand_uniform(&localRandState)),
                                           0.5f * (1.0f + curand_uniform(&localRandState)),
                                           0.5f * (1.0f + curand_uniform(&localRandState)));
                        float fuzz = 0.5f * curand_uniform(&localRandState);
                        sphereMat = new metal(albedo, fuzz);
                        *(d_list + cnt++) = new sphere(center, 0.2f, sphereMat);
                    } else {
                        // glass
                        sphereMat = new dielectric(1.5f);
                        *(d_list + cnt++) = new sphere(center, 0.2f, sphereMat);
                    }
            }
        }
        material *mat1 = new dielectric(1.5f);
        *(d_list+ cnt++) = new moving_sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, mat1);

        material *mat2 = new lambertian(vec3(0.4f, 0.2f, 0.1f), false);
        *(d_list+ cnt++) = new moving_sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, mat2);

        material *mat3 = new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f);
        *(d_list+ cnt++) = new moving_sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, mat3);

        material *mat4 = new lambertian(vec3(0.4f, 0.2f, 0.1f), true);
        *(d_list+ cnt++) = new moving_sphere(vec3(6.5f, 0.3f, 2.0f), 0.3f, mat4);

        // Camera
        vec3 lookFrom(13.0f, 2.0f, 3.0f);
        vec3 lookAt(0.0f, 0.0f, 0.0f);
        vec3 vUp(0.0f, 1.0f, 0.0f);

        float distToFocus = 10.0f;
        float aperture = 0.1f;
        float aspect_ratio = 3.0f / 2.0f;

        *d_cam = new camera(lookFrom, lookAt, vUp, 20.0f, aspect_ratio, aperture, distToFocus, 0.0, 1.0);
    }
}

__global__ void freeWorld(hittable **d_list, hittable **d_world, camera **d_cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < SPHERE_COUNT; i ++) {
            delete ((moving_sphere *)d_list[i])->mat_ptr;
            delete *(d_list + i);
        }
        delete *(d_world);
        delete *(d_cam);
    }
}

int main(void) {
    int nx = 1200;
    int ny = 800;
    int ns = 50;

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
    cudaStatus = cudaMalloc((void**)&d_list, SPHERE_COUNT * sizeof(hittable*));
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
    std::ofstream ppmFile("moving_spheres.ppm");

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