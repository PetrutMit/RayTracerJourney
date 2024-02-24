#ifndef AABB_CUH
#define AABB_CUH

#include "vec3.cuh"
#include "ray.cuh"
#include "interval.cuh"

class aabb {
    public:
        interval x, y, z;

        __device__ aabb() {}
        __device__ aabb(const interval& x, const interval& y, const interval& z) : x(x), y(y), z(z) {}
        __device__ aabb(const vec3& a, const vec3& b) {
            x = interval(fminf(a.x(), b.x()), fmaxf(a.x(), b.x()));
            y = interval(fminf(a.y(), b.y()), fmaxf(a.y(), b.y()));
            z = interval(fminf(a.z(), b.z()), fmaxf(a.z(), b.z()));
        }
        __device__ aabb(const aabb &box0, const aabb &box1) {
            x = interval(box0.x, box1.x);
            y = interval(box0.y, box1.y);
            z = interval(box0.z, box1.z);
        }

        __device__ const interval& axis(int n) const {
            switch (n) {
                case 0:
                    return x;
                case 1:
                    return y;
                case 2:
                    return z;
            }
        }

        __device__ bool hit(const ray& r, interval ray_t) const {
            for (int a = 0; a < 3; a++) {
                float invD = 1.0f / r.direction()[a];

                float t0 = (axis(a).min - r.origin()[a])* invD; 
                float t1 = (axis(a).max - r.origin()[a])* invD;

                if (invD < 0.0f) {
                    float temp = t0;
                    t0 = t1;
                    t1 = temp;
                }

                if (t0 > ray_t.min) { 
                    ray_t.min = t0;
                }
                if (t1 < ray_t.max) {
                    ray_t.max = t1;
                }

                if (ray_t.max <= ray_t.min) {
                    return false;
                }
            }
            return true;
        }

        __device__ aabb pad() {
            float eps = 0.0001f;
            interval new_x (x.size() >= eps ? x : x.expand(eps));
            interval new_y (y.size() >= eps ? y : y.expand(eps));
            interval new_z (z.size() >= eps ? z : z.expand(eps));
            return aabb(new_x, new_y, new_z);
        }
};

#ifdef __CUDACC__
__device__ aabb operator+(const aabb &bbox, const vec3 &offset) {
    return aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

__device__ aabb operator+(const vec3 &offset, const aabb &bbox) {
    return bbox + offset;
}
#endif

#endif
