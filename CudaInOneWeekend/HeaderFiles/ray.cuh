#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class ray {
public:
    __device__ ray() {}
    __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    __device__ vec3 origin() const { return orig; }
    __device__ vec3 direction() const { return dir; }

    __device__ vec3 at(double t) const {
        return orig + t * dir;
    }

    point3 orig;
    vec3 dir;
};

#endif