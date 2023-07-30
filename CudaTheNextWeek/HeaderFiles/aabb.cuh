#ifndef AABB_CUH
#define AABB_CUH

#include "vec3.cuh"
#include "ray.cuh"

class aabb {
    public:
        __device__ aabb() {}
        __device__ aabb(const vec3& a, const vec3& b) {minimum = a; maximum = b;}

        __device__ vec3 min() const {return minimum;}
        __device__ vec3 max() const {return maximum;}

        __device__ bool hit(const ray& r, float tmin, float tmax) const {
            for (int a = 0; a < 3; a++) {
                float invD = 1.0f / r.direction()[a];
                float t0 = (min()[a] - r.origin()[a]) * invD;
                float t1 = (max()[a] - r.origin()[a]) * invD;
                if (invD < 0.0f) {
                    float temp = t0;
                    t0 = t1;
                    t1 = temp;
                }
                tmin = t0 > tmin ? t0 : tmin;
                tmax = t1 < tmax ? t1 : tmax;
                if (tmax <= tmin) return false;
            }
            return true;
        }

        vec3 minimum;
        vec3 maximum;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
    vec3 small(fminf(box0.min().x(), box1.min().x()),
               fminf(box0.min().y(), box1.min().y()),
               fminf(box0.min().z(), box1.min().z()));
    vec3 big(fmaxf(box0.max().x(), box1.max().x()),
             fmaxf(box0.max().y(), box1.max().y()),
             fmaxf(box0.max().z(), box1.max().z()));

    return aabb(small, big);
}

#endif
