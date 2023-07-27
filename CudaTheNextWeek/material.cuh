#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "ray.cuh"
#include "hittable.cuh"
#include "texture.cuh"

#include "random.cuh"

struct hit_record;

class material {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *localRandState) const = 0;

        __device__ virtual color emitted(float u, float v, const point3& p) const {
            return color(0.0f, 0.0f, 0.0f);
        }
};

class lambertian : public material {
    public:
        __device__ lambertian(const color& a) : albedo(new constant_texture(a)) {}
        __device__ lambertian(my_texture *a) : albedo(a) {}

        __device__ ~lambertian() {
            delete albedo;
        }

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *localRandState) const override {
            vec3 scatter_direction = rec.normal + randomUnitVector(localRandState);
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = ray(rec.p, scatter_direction, r_in.time());
            attenuation = albedo->value(rec.u, rec.v, rec.p);
            return true;
        }

    public:
        my_texture *albedo;
};

class metal : public material {
    public:
        __device__ metal(const color& a, float f) : albedo(a), fuzz(f) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *localRandState) const override {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz * randomInUnitSphere(localRandState), r_in.time());
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

    public:
        color albedo;
        float fuzz;
};


class dielectric : public material {
    public:
        __device__ dielectric(float ri) : ref_idx(ri) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *localRandState) const override {
            attenuation = color(1.0f, 1.0f, 1.0f);
            float etai_over_etat = rec.front_face ? (1.0f / ref_idx) : ref_idx;

            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

            bool cannotRefract = etai_over_etat * sin_theta > 1.0f;
            vec3 direction;

            if (cannotRefract || reflectance(cos_theta, etai_over_etat) > curand_uniform(localRandState))
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, etai_over_etat);

            scattered = ray(rec.p, direction, r_in.time());
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

class diffuse_light : public material {
    public:
        __device__ diffuse_light(my_texture *a) : emit(a) {}
        __device__ diffuse_light(color c) : emit(new constant_texture(c)) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *localRandState) const override {
            return false;
        }

        __device__ virtual color emitted(float u, float v, const point3& p) const override {
            return emit->value(u, v, p);
        }

    public:
        my_texture *emit;
};

class isotropic : public material {
    public:
        __device__ isotropic(color c) : albedo(new constant_texture(c)) {}
        __device__ isotropic(my_texture *a) : albedo(a) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *localRandState) const override {
            scattered = ray(rec.p, randomInUnitSphere(localRandState), r_in.time());
            attenuation = albedo->value(rec.u, rec.v, rec.p);
            return true;
        }
    public:
        my_texture *albedo;
};

#endif
