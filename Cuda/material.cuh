#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "ray.cuh"

#include <curand_kernel.h>

// Random auxilliary functions
// Make a random vector in all directions
__device__ vec3 randomVector(curandState *localRandState) {
    float a = curand_uniform(localRandState);
    float b = curand_uniform(localRandState);
    float c = curand_uniform(localRandState);
    return unit_vector(vec3(a,b,c));
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

struct hit_record;

class material {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *localRandState) const = 0;
};

class lambertian : public material {
    public:
        __device__ lambertian(const color& a) : albedo(a) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *localRandState) const override {
            vec3 scatter_direction = rec.normal + randomUnitVector(localRandState);
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;
            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo;
            return true;
        }

    public:
        color albedo;
};

class metal : public material {
    public:
        __device__ metal(const color& a, float f) : albedo(a), fuzz(f) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *localRandState) const override {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz * randomInUnitSphere(localRandState));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

    public:
        color albedo;
        float fuzz;
};

#endif
