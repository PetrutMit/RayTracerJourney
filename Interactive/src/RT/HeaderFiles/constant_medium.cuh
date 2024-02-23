#ifndef CONSTANT_MEDIUM_CUH
#define CONSTANT_MEDIUM_CUH

#include "hittable.cuh"
#include "material.cuh"
#include "texture.cuh"

#include "random.cuh"
#include "aabb.cuh"
#include "vec3.cuh"

class constant_medium : public hittable {
    public:
        __device__ constant_medium(hittable *b, float d, my_texture *a, curandState *state) : boundary(b), neg_inv_density(-1.0f/d), phase_function(new isotropic(a)), rand_state(state) {}
        __device__ constant_medium(hittable *b, float d, color c, curandState *state) : boundary(b), neg_inv_density(-1/d), phase_function(new isotropic(c)), rand_state(state) {}
        __device__ ~constant_medium() {
            delete boundary;
            delete phase_function;
        }

        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
        __device__ virtual aabb bounding_box() const override;

    public:
        hittable *boundary;
        float neg_inv_density;
        material *phase_function;
        curandState *rand_state;
};

__device__ bool constant_medium::hit(const ray& r, interval ray_t, hit_record& rec) const {
    hit_record rec1, rec2;

    if (!boundary->hit(r, interval(-INF, INF), rec1))
        return false;

    if (!boundary->hit(r, interval(rec1.t + 0.001, INF), rec2))
        return false;

    if (rec1.t < ray_t.min)
        rec1.t = ray_t.min;

    if (rec2.t > ray_t.max)
        rec2.t = ray_t.max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    const float ray_length = r.direction().length();
    const float distance_inside_boundary = (rec2.t - rec1.t) * ray_length;

    //const float hit_distance = neg_inv_density * log(randomFloat(rand_state, 0, 1));
    const float hit_distance = neg_inv_density * logf(curand_uniform(rand_state));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    rec.normal = vec3(1, 0, 0); // arbitrary
    rec.front_face = true; // also arbitrary
    rec.mat_ptr = phase_function;

    return true;
}

__device__ aabb constant_medium::bounding_box() const {
    return boundary->bounding_box();
}

#endif

