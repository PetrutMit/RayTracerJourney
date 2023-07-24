#ifndef MOVING_SPHERE_CUH
#define MOVING_SPHERE_CUH

#include "hittable.cuh"
#include "vec3.cuh"
#include "material.cuh"
#include "aabb.cuh"

#define PI 3.141592653589793f

class moving_sphere : public hittable {
    public:
        __device__ moving_sphere() {}
        __device__ moving_sphere(point3 cen0, float r, material *m)
                : center0(cen0), center1(cen0), time0(0.0), time1(1.0), radius(r), mat_ptr(m)
            {};
        __device__ moving_sphere(point3 cen0, float r, material *m, float t0, float t1, point3 cen1)
                : center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m)
            {};

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;
        __device__ point3 center(float time) const;

    private:
        __device__ static void get_sphere_uv(const vec3& p, float& u, float& v) {
            float phi = atan2(p.z(), p.x());
            float theta = asin(p.y());
            u = 1-(phi + PI) / (2*PI);
            v = (theta + PI/2) / PI;
        }

    public:
        point3 center0, center1;
        float time0, time1;
        float radius;
        material *mat_ptr;
};

__device__ point3 moving_sphere::center(float tm) const {
    return center0 + ((tm - time0) / (time1 - time0)) * (center1 - center0);
}

__device__ bool moving_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center(r.time());
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius*radius;

    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;

    return true;
}

__device__ bool moving_sphere::bounding_box(float time0, float time1, aabb& output_box) const {
    aabb box0(
        center(time0) - vec3(radius, radius, radius),
        center(time0) + vec3(radius, radius, radius));
    if (center(time0) == center(time1)) {
        output_box = box0;
        return true;
    }

    aabb box1(
        center(time1) - vec3(radius, radius, radius),
        center(time1) + vec3(radius, radius, radius));
    output_box = surrounding_box(box0, box1);
    return true;
}

#endif