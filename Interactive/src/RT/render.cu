#include "HeaderFiles/header.cuh"

#include "HeaderFiles/random.cuh"
#include "HeaderFiles/bvh.cuh"

#include "HeaderFiles/render.cuh"

#define NODE_COUNT 8

__device__ color Render::rayColor(const ray& r, const color& background, hittable **world, curandState *localRandState) {
    ray curRay = r;

    color curAttenuation(1.0f, 1.0f, 1.0f);
    color curEmitted(0.0f, 0.0f, 0.0f);
    hit_record rec;

    for (int i = 0; i < 20; i ++) {
        if ((*world)->hit(curRay, interval(0.001, INF), rec)) {
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

__global__ void Render::renderInit(int maxX, int maxY, curandState *randStatePixels, curandState *randStateWorld) {
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

__global__ void Render::raytrace(vec3 *fb, int maxX, int maxY, int ns, camera **cam, hittable **world, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return;

    int pixelIndex = j * maxX + i;
    curandState localRandState = randState[pixelIndex];
    color pixelColor(0.0f, 0.0f, 0.0f);
    color background(0.05f, 0.05f, 0.05f);
    for (int s = 0; s < ns; s ++) {
        float u = float(i + curand_uniform(&localRandState)) / float(maxX);
        float v = float(j + curand_uniform(&localRandState)) / float(maxY);

        ray r = (*cam)->get_ray(u, v, &localRandState);
        pixelColor += rayColor(r, background, world, &localRandState);
    }

    getColor(pixelColor, ns);

    fb[pixelIndex] = pixelColor;
}

__global__ void Render::allocateWorld(hittable **d_list, hittable **d_world, camera **d_cam, curandState *randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        lambertian *ground = new lambertian(color(0.48f, 0.83f, 0.53f));

        int boxes_per_side = 4;

        hittable **boxes = new hittable*[boxes_per_side * boxes_per_side];
        int cnt = 0;

        for (int i = 0; i < boxes_per_side; i ++) {
            for (int j = 0; j < boxes_per_side; j ++) {
                float w = 200.0f;
                float x0 = i * w;
                float z0 = j * w;
                float y0 = 0.0f;
                float x1 = x0 + w;
                float y1 = randomFloat(randState, 1.0f, 101.0f);
                float z1 = z0 + w;

                boxes[cnt ++] = box(vec3(x0, y0, z0), vec3(x1, y1, z1), ground);
            }
        }

        if (useBVH) {
            *(d_list) = new bvh_node(boxes, cnt, randState);
        } else {
            *(d_list) = new hittable_list(boxes, cnt);
        }

        diffuse_light *light = new diffuse_light(color(7.0f, 7.0f, 7.0f));
        quad *light_shape = new quad(vec3(123.0f, 554.0f, 147.0f), vec3(300.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 265.0f), light);

        *(d_list + 1) = light_shape;

        vec3 center1 = vec3(400.0f, 400.0f, 200.0f);
        vec3 center2 = center1 + vec3(30.0f, 0.0f, 0.0f);

        *(d_list + 2) = new sphere(center1, center2, 50.0f, new lambertian(color(0.7f, 0.3f, 0.1f)));
        *(d_list + 3) = new sphere(vec3(260.0f, 150.0f, 45.0f), 50.0f, new dielectric(1.5f));
        *(d_list + 4) = new sphere(vec3(400.0f, 200.0f, 400.0f), 100.0f, new metal(color(0.8f, 0.8f, 0.9f), 10.0f));

        *(d_list + 5) = new sphere(vec3(360.0f, 150.0f, 145.0f), 70.0f, new metal(color(0.3f, 0.8f, 0.2f), 0.2f));

        noise_texture *pertext = new noise_texture(randState, 0.1f);
        *(d_list + 6) = new sphere(vec3(220.0f, 280.0f, 300.0f), 80.0f, new lambertian(pertext));

        hittable **spheres = new hittable*[1024];
        lambertian *blue = new lambertian(color(0.2f, 0.2f, 0.7f));
        for (int i = 0; i < 1024; i ++) {
            spheres[i] = new sphere(vec3(randomVectorBetween(randState, 0.0f, 165.0f)), 10.0f, blue);
        }

        if (useBVH) {
             *(d_list + 7) = new translate(new rotate_y(new bvh_node(spheres, 1024, randState), 15.0f), vec3(-100.0f, 270.0f, 395.0f));
         } else {
             *(d_list + 7) = new translate(new rotate_y(new hittable_list(spheres, 1024), 15.0f), vec3(-100.0f, 270.0f, 395.0f));
        }

        *(d_world) = new hittable_list(d_list, NODE_COUNT);

        vec3 lookFrom(478.0f, 278.0f, -600.0f);
        vec3 lookAt(278.0f, 278.0f, 0.0f);
        float distToFocus = 10.0f;
        float aperture = 0.0f;
        float aspect_ratio = 3.0f / 2.0f;
        float vfov = 40.0f;

        *(d_cam) = new camera(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), vfov, aspect_ratio, aperture, distToFocus, 0.0f, 1.0f);
    }
}

__global__ void Render::freeWorld(hittable **d_list, hittable **d_world, camera **d_cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < NODE_COUNT; i++) {
            delete *(d_list + i);
        }
        delete *(d_world);
        delete *(d_cam);
    }
}

Render::Render(GLuint nx, GLuint ny, cudaGraphicsResource_t cuda_pbo_resource) {
    _nx = nx;
    _ny = ny;
    _cuda_pbo_resource = cuda_pbo_resource;

    GLint num_pixels = _nx * _ny;

    cudaError_t cudaStatus;

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
    cudaStatus = cudaMalloc((void**)&d_list, NODE_COUNT * sizeof(hittable*));
    checkReturn(cudaStatus);

    hittable **d_world;
    cudaStatus = cudaMalloc((void**)&d_world, sizeof(hittable*));
    checkReturn(cudaStatus);

    // create camera
    camera **d_cam;
    cudaStatus = cudaMalloc((void**)&d_cam, sizeof(camera*));
    checkReturn(cudaStatus);

    dim3 blockCount(_nx + TX - 1 / TX, _ny + TY - 1 / TY);
    dim3 blockSize(TX, TY);

    renderInit<<<blockCount, blockSize>>>(_nx, _ny, d_randState, d_worldRandState);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    allocateWorld<<<1, 1>>>(d_list, d_world, d_cam, d_worldRandState);
    checkReturn(cudaGetLastError());

    _d_cam = d_cam;
    _d_world = d_world;
    _d_randStatePixels = d_randState;

    cudaFree(d_worldRandState);
    cudaFree(d_list);
}

__host__ void Render::render() {
    dim3 blockCount(_nx + TX - 1 / TX, _ny + TY - 1 / TY);
    dim3 blockSize(TX, TY);
    GLuint ns = 5;

    // Get the pointer to the frame buffer
    vec3 *fb;
    size_t size;
    cudaError_t cudaStatus;
    
    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&fb, &size, _cuda_pbo_resource);
    checkReturn(cudaStatus);
    
    raytrace<<<blockCount, blockSize>>>(fb, _nx, _ny, ns, _d_cam, _d_world, _d_randStatePixels);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    // Unmap the resource
    cudaStatus = cudaGraphicsUnmapResources(1, &_cuda_pbo_resource);
    checkReturn(cudaStatus);
}

__host__ void Render::free() {
    cudaError_t cudaStatus;

    // free world of hittable objects
    freeWorld<<<1, 1>>(_d_list, _d_world, _d_cam);
    checkReturn(cudaGetLastError());

    cudaStatus = cudaFree(_d_list);
    checkReturn(cudaStatus);

    cudaStatus = cudaFree(_d_world);
    checkReturn(cudaStatus);
}
