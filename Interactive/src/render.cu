#include "HeaderFiles/header.cuh"

#include "HeaderFiles/random.cuh"
#include "HeaderFiles/bvh.cuh"

#include "HeaderFiles/render.cuh"

#include <glm/packing.hpp>

#define NODE_COUNT 8
#define FILTER_SIZE 4

__device__ color rayColor(const ray& r, const color& background, hittable_list **world, curandState *localRandState, GBufferTexel *gBuffer) {
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

                // Populate gBuffer on depth 0
                if (i == 0) {
                    gBuffer->normal = rec.normal;
                    gBuffer->position = rec.p;
                }
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

__global__ void raytrace(vec3 *fbColor, int maxX, int maxY, camera **cam, hittable_list **world,
                         curandState *randState, int frameCount, float deltaTime,
                          float camera_X, float camera_Y, float camera_Z,
                         GBufferTexel *gBuffer, uint32_t *fbOut, bool denoise) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return;

    // Adjust our camera view
    if (i == 0 && j == 0) {
        (*cam)->adjust_position(vec3(camera_X, camera_Y, camera_Z));
    }

    int pixelIndex = j * maxX + i;
    curandState localRandState = randState[pixelIndex];
    color pixelColor(0.0f, 0.0f, 0.0f);
    color background(0.2f, 0.2f, 0.2f);

    // Just one sample per pixel
    float u = float(i + curand_uniform(&localRandState)) / float(maxX);
    float v = float(j + curand_uniform(&localRandState)) / float(maxY);

    ray r = (*cam)->get_ray(u, v, &localRandState);
    pixelColor = rayColor(r, background, world, &localRandState, &gBuffer[pixelIndex]);

    getColor(pixelColor, 1);

    // Frame accumulation in fbColor
    fbColor[pixelIndex] = mix(fbColor[pixelIndex], pixelColor, 0.3);

    if ( !denoise ) {
        fbOut[pixelIndex] = glm::packUnorm4x8(glm::vec4(pixelColor.z(), pixelColor.y(), pixelColor.x(), 1));
	}
    
   // Change rand state
    curand_init(1984 + pixelIndex + frameCount, 0, 0, &randState[pixelIndex]);

}

__global__ void atrousDenoise(GBufferTexel* gBuffer, int maxX, int maxY, int stepWidth, vec3* rayTracedInput, vec3* denoisedOutput, uint32_t* fb) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return;

    const float c_phi = 2.45f;
    const float n_phi = 2.30f;
    const float p_phi = 2.25f;

    static constexpr float kernel[] = { 3.f / 8.f, 1.f / 4.f, 1.f / 16.f };

    int pixelIndex = j * maxX + i;

    GBufferTexel center = gBuffer[pixelIndex];
    vec3 center_normal = center.normal;
    vec3 center_position = center.position;
    vec3 center_albedo = rayTracedInput[pixelIndex];

    vec3 sum_albedo(0.0f);
    float sum_weight = 0.0f;
    int dx, dy, u, v, index, kernel_index, dist;
    vec3 normal, position, albedo, diff;
    float p_weight, n_weight, c_weight, weight;

    for (dy = -2; dy <= 2; dy++) {
        for (dx = -2; dx <= 2; dx++) {
            u = glm::clamp(i + dx * stepWidth, 0, maxX);
            v = glm::clamp(j + dy * stepWidth, 0, maxY);

            index = v * maxX + u;

            normal = gBuffer[index].normal;
            position = gBuffer[index].position;
            albedo = rayTracedInput[index];

            diff = center_position - position;
            dist = diff.length_squared();
            p_weight = fminf(std::exp(-dist / p_phi), 1.0f);

            diff = center_normal - normal;
            dist = fmaxf(diff.length_squared() / static_cast<float>(stepWidth * stepWidth), 0.0);
            n_weight = fminf(std::exp(-dist / n_phi), 1.0f);

            diff = center_albedo - albedo;
            dist = diff.length_squared();
            c_weight = fminf(std::exp(-dist / c_phi), 1.0f);

            weight = p_weight * n_weight * c_weight;

            kernel_index = fminf(std::abs(dx), std::abs(dy));
            sum_albedo += albedo * kernel[kernel_index] * weight;

            sum_weight += kernel[kernel_index] * weight;
        }
    }

    vec3 denoisedPixel = sum_albedo / sum_weight;
    denoisedOutput[pixelIndex] = denoisedPixel;
    // On the last iteration, write the pixel to OpenGL bound buffer
    if (stepWidth == FILTER_SIZE) {
        fb[pixelIndex] = glm::packUnorm4x8(glm::vec4(denoisedPixel.z(), denoisedPixel.y(), denoisedPixel.x(), 1.0f));
    }
}

__global__ void allocateWorld(int maxX, int maxY, hittable **d_list, hittable_list **d_world, camera **d_cam, curandState *randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        lambertian *ground = new lambertian(color(0.1f, 0.3f, 0.3f));
        *(d_list) = new sphere(vec3(0.0f, -2000.0f, 0.0f), 2000.0f, ground);

        diffuse_light *light = new diffuse_light(color(1.0f, 1.0f, 1.0f));
        sphere* moon = new sphere(vec3(-550.0f, 350.0f, 550.0f), 50.0f, light);
        *(d_list + 1) = moon; 

        *(d_list + 2) = new sphere(vec3(-80.0f, 20.0f, -300.0f), 40.0f, new metal(color(0.1f, 0.9f, 0.1f), 0.1f));
 
        hittable **spheres = new hittable*[64];
        lambertian *red = new lambertian(color(0.9f, 0.2f, 0.1f));
        for (int i = 0; i < 64; i ++) {
            spheres[i] = new sphere(vec3(randomVectorBetween(randState, -50.0f, 0.0f) + vec3(50.0f, 30.0f, -300.0f)), 7.0f, red);
        }

        *(d_list + 3) = new bvh_node(spheres, 64, randState);

        // Create some random height boxes
        hittable **boxes = new hittable*[4];

        float width = 20.0f;
        int choice;
        float x0, z0, x1, z1, y0, y1;
        material *mat;
        for (int i = 0; i < 4; i ++) {
            x0 = 80.0f + i * width;
            z0 = -300.0f;

            x1 = x0 + width;
            z1 = -300.0f + width;
    
            y0 = -30.0f;
            y1 = randomFloat(randState, 1.0f, 100.0f);
            
            choice = randomInt(randState, 0, 2);
            switch (choice)
            {
            case 0:
                mat = new lambertian(randomVector(randState));
                break;
            case 1:
                mat = new metal(randomVector(randState), randomFloat(randState, 0.0f, 0.3f));
                break;
            }
            boxes[i] = box(vec3(x0, y0, z0), vec3(x1, y1, z1), mat);
		}

        *(d_list + 4) = new bvh_node(boxes, 4, randState);

        // Until here, we have 1 + 1 + 1 + 64 + 24 = 91 objects

        hittable** spheres2 = new hittable *[64];
        lambertian* blue = new lambertian(color(0.1f, 0.1f, 0.9f));
        for (int i = 0; i < 64; i++) {
            spheres2[i] = new sphere(vec3(randomVectorBetween(randState, -50.0f, 0.0f) + vec3(-150.0f, 30.0f, -300.0f)), 7.0f, blue);
        }
        *(d_list + 5) = new bvh_node(spheres2, 64, randState);

   
        // Create a cone of spheres
        hittable **groundSpheres = new hittable*[64];
        vec3 center(250.0f, -10.0f, -300.0f);
        float sphereRadius = 10.0f;
        int cnt = 0;

        // Layers of spheres
        for (int i = 0; i < 5; i++) {
            float circleRadius = 10.0f * (i + 1);
            for (int j = 0; j < 4 * (i + 1); j++) {
				float angle = 2.0f * PI * j / (4.0f * (i + 1));
				float x = circleRadius * cos(angle);
				float z = circleRadius * sin(angle);
				float y = 20.0f * i;
				vec3 position = vec3(x, y, z) + center;
				groundSpheres[cnt ++] = new sphere(position, sphereRadius, new lambertian(randomVector(randState)));
			}
    	}
        // 4 + 8 + 12 + 16 + 20 = 60 spheres
        // add 4 more
        groundSpheres[cnt++] = new sphere(vec3(250.0f, 100.0f, -300.0f), 10.0f, new lambertian(vec3(0.8f, 0.8f, 0.8f)));
        groundSpheres[cnt++] = new sphere(vec3(250.0f, 120.0f, -300.0f), 10.0f, new lambertian(vec3(0.8f, 0.8f, 0.8f)));
        groundSpheres[cnt++] = new sphere(vec3(250.0f, 140.0f, -300.0f), 10.0f, new lambertian(vec3(0.8f, 0.8f, 0.8f)));
        groundSpheres[cnt++] = new sphere(vec3(250.0f, 160.0f, -300.0f), 10.0f, new lambertian(vec3(0.8f, 0.8f, 0.8f)));
        *(d_list + 6) = new bvh_node(groundSpheres, 64, randState);

        *(d_list + 7) = new sphere(vec3(-250.0f, 0.0f, -270.0f), 40.0f, new dielectric(1.5f));

        
        *(d_world) = new hittable_list(d_list, NODE_COUNT);

        vec3 lookFrom(0.0f, 0.0f, -600.0f);
        vec3 lookAt(0.0f, 0.0f, 0.0f);
        float distToFocus = 10.0f;
        float aperture = 0.0f;
        float aspect_ratio = static_cast<float>(maxX) / static_cast<float>(maxY);
        float vfov = 40.0f;

        *(d_cam) = new camera(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), vfov, aspect_ratio, aperture, distToFocus);
    }
}

__global__ void freeWorld(hittable **d_list, hittable_list **d_world, camera **d_cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < NODE_COUNT; i++) {
            delete *(d_list + i);
        }
        delete *(d_world);
        delete *(d_cam);
    }
}

Render::Render(int nx, int ny, cudaGraphicsResource_t cuda_pbo_resource) {
    _nx = nx;
    _ny = ny;

    int num_pixels = _nx * _ny;

    cudaError_t cudaStatus;

    // create random state for each pixel
    curandState *d_randStatePixels;
    cudaStatus = cudaMalloc((void**)&d_randStatePixels, num_pixels * sizeof(curandState));
    checkReturn(cudaStatus);

    // create random state for world construction
    curandState *d_randStateWorld;
    cudaStatus = cudaMalloc((void**)&d_randStateWorld, sizeof(curandState));
    checkReturn(cudaStatus);

    // create world of hittable objects
    hittable **d_list;
    cudaStatus = cudaMalloc((void**)&d_list, NODE_COUNT * sizeof(hittable*));
    checkReturn(cudaStatus);

    hittable_list **d_world;
    cudaStatus = cudaMalloc((void**)&d_world, sizeof(hittable_list*));
    checkReturn(cudaStatus);

    // create camera
    camera **d_cam;
    cudaStatus = cudaMalloc((void**)&d_cam, sizeof(camera*));
    checkReturn(cudaStatus);

    // create gBuffer
    GBufferTexel *d_gbuffer;
    cudaStatus = cudaMalloc((void**)&d_gbuffer, num_pixels * sizeof(GBufferTexel));
    checkReturn(cudaStatus);

    // create output color for raytracing
    cudaStatus = cudaMalloc((void**)&_d_rayTracedImage, num_pixels * sizeof(vec3));
    checkReturn(cudaStatus);

    // create output color for denoising
    cudaStatus = cudaMalloc((void**)&_d_denoisedImage, num_pixels * sizeof(vec3));
    checkReturn(cudaStatus);

    // Even though we have an iterative approach, we still need a bigger stack
    size_t size;
    cudaDeviceGetLimit(&size, cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize, 2 * size);

    dim3 blockCount(_nx + TX - 1 / TX, _ny + TY - 1 / TY);
    dim3 blockSize(TX, TY);

    renderInit<<<blockCount, blockSize>>>(_nx, _ny, d_randStatePixels, d_randStateWorld);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    allocateWorld<<<1, 1>>>(_nx, _ny, d_list, d_world, d_cam, d_randStateWorld);
    checkReturn(cudaGetLastError());

    // Free random state for world construction
    cudaFree(d_randStateWorld);

    _d_cam = d_cam;
    _d_world = d_world;
    _d_randStatePixels = d_randStatePixels;
    _d_gBuffer = d_gbuffer;

    _cuda_pbo_resource = cuda_pbo_resource;
    cudaStatus = cudaGraphicsMapResources(1, &_cuda_pbo_resource);
    checkReturn(cudaStatus);

    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&_d_output, NULL, _cuda_pbo_resource);
    checkReturn(cudaStatus);
}

__host__ void Render::render(int frameCount, float deltaTime, float camera_X, float camera_Y, float camera_Z, bool denoise) {
    const dim3 blockCount(_nx + TX - 1 / TX, _ny + TY - 1 / TY);
    const dim3 blockSize(TX, TY);

    raytrace << <blockCount, blockSize >> > (_d_rayTracedImage, _nx, _ny, _d_cam, _d_world,
        _d_randStatePixels, frameCount, deltaTime, camera_X, camera_Y, camera_Z, _d_gBuffer, _d_output, denoise);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());
}

__host__ void Render::denoise() {
    const dim3 blockCount(_nx + TX - 1 / TX, _ny + TY - 1 / TY);
    const dim3 blockSize(TX, TY);

    for (int stepWidth = 1; stepWidth <= FILTER_SIZE; stepWidth *= 2) {
        atrousDenoise<<<blockCount, blockSize>>>(_d_gBuffer, _nx, _ny, stepWidth, _d_rayTracedImage, _d_denoisedImage, _d_output);
        checkReturn(cudaGetLastError());
        checkReturn(cudaDeviceSynchronize());
        // Swap the buffers
        vec3 *temp = _d_rayTracedImage;
        _d_rayTracedImage = _d_denoisedImage;
        _d_denoisedImage = temp;
    }
}

Render::~Render() {
    freeWorld<<<1, 1>>>((*_d_world)->list, _d_world, _d_cam);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    cudaFree(_d_randStatePixels);
    checkReturn(cudaGraphicsUnmapResources(1, &_cuda_pbo_resource));
}
