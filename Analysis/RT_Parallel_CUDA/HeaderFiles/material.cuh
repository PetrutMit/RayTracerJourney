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
        __device__ lambertian(const color& a, const bool light) : albedo(a), isLight(light) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *localRandState) const override {
            vec3 scatter_direction = rec.normal + randomUnitVector(localRandState);
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;
                scattered = ray(rec.p, scatter_direction);

            attenuation = albedo;
            if (isLight) {
                attenuation = attenuation * 10;
            }
            return true;
        }

    public:
        color albedo;
        bool isLight;
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

#define min(a,b) ((a) < (b) ? (a) : (b))

class dielectric : public material {
    public:
        __device__ dielectric(float ri) : ref_idx(ri) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *localRandState) const override {
            attenuation = color(1.0f, 1.0f, 1.0f);
            float etai_over_etat = rec.front_face ? (1.0f / ref_idx) : ref_idx;

            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = min(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

            bool cannotRefract = etai_over_etat * sin_theta > 1.0f;
            vec3 direction;

            if (cannotRefract || reflectance(cos_theta, etai_over_etat) > curand_uniform(localRandState))
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, etai_over_etat);

            scattered = ray(rec.p, direction);
            return true;
        }

    public:
        float ref_idx;

    private:
        __device__ static float reflectance(float cosine, float ref_idx) {
            // Use Schlick's approximation for reflectance
            float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
            r0 = r0 * r0;
            return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
        }
};

#endif
