#ifndef TRANSLATE_CUH
#define TRANSLATE_CUH

#include "hittable.cuh"
#include "vec3.cuh"
#include "ray.cuh"

class translate : public hittable {
    public:
        __device__ translate(hittable *p, const vec3& displacement) : ptr(p), offset(displacement) {
            bbox = p->bounding_box() + offset;
        }

        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
        __device__ virtual aabb bounding_box() const override;

    public:
        hittable *ptr;
        vec3 offset;
        aabb bbox;
};

__device__ bool translate::hit(const ray& r, interval ray_t, hit_record& rec) const {
    // This line is problematic
    ray moved_r = ray(r.origin() - offset, r.direction(), r.time());

    if (!ptr->hit(moved_r, ray_t, rec))
        return false;
    rec.p += offset;

    return true;
}

__device__ aabb translate::bounding_box() const {
    return bbox;
}

#endif
