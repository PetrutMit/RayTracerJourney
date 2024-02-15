#ifndef RANDOM_CUH
#define RANDOM_CUH

#include "vec3.cuh"
#include "curand_kernel.h"

__device__ float randomFloat(curandState *state, float min, float max) {
    return min + (max - min) * curand_uniform(state);
}

__device__ int randomInt(curandState *state, int min, int max) {
    return (int)randomFloat(state, min, max);
}

__device__ vec3 randomVector(curandState *localRandState) {
    float a = curand_uniform(localRandState);
    float b = curand_uniform(localRandState);
    float c = curand_uniform(localRandState);
    return unit_vector(vec3(a,b,c));
}

__device__ vec3 randomVectorBetween(curandState *localRandState, float min, float max) {
    float a = randomFloat(localRandState, min, max);
    float b = randomFloat(localRandState, min, max);
    float c = randomFloat(localRandState, min, max);
    return vec3(a,b,c);
}

__device__ vec3 randomInUnitDisk(curandState *state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(state), curand_uniform(state), 0.0f) - vec3(1.0f, 1.0f, 0.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

__device__ vec3 randomInUnitSphere(curandState *localRandState) {
    vec3 p;
    do {
        p = 2.0f * randomVector(localRandState) - vec3(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ vec3 randomUnitVector(curandState *localRandState) {
    return unit_vector(randomInUnitSphere(localRandState));
}

__device__ vec3 randomInHemiSphere(const vec3& normal, curandState *localRandState) {
    vec3 inUnitSphere = randomInUnitSphere(localRandState);
    if (dot(inUnitSphere, normal) > 0.0f) {
        return inUnitSphere;
    } else {
        return -inUnitSphere;
    }
}

#endif