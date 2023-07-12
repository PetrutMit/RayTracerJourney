#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "hittable.cuh"
#include "vec3.cuh"
#include "material.cuh"

class sphere : public hittable {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 cen, float r , material *mat) : center(cen), radius(r), mat_ptr(mat)  {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override;

    public:
        vec3 center;
        float radius;
        material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float half_b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;

    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (root < tmin || tmax < root) {
        root = (-half_b + sqrtd) / a;
        if (root < tmin || tmax < root) return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

#endif