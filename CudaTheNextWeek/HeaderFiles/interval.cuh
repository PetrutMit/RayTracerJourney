#ifndef INTERVAL_H
#define INTERVAL_H

#include "vec3.cuh"

#define INF 0x7f8000

class interval {
    public:
        float min, max;

        __device__ interval() : min(INF), max(-INF) {}
        __device__ interval(float min, float max) : min(min), max(max) {}
        __device__ interval(const interval &a, const interval &b) : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

        __device__ bool contains(float x) const {
            return (x >= min && x <= max);
        }
        __device__ bool surrounds(float x) const {
            return (x > min && x < max);
        }
        __device__ interval expand(float x) const {
            float half = x / 2;
            return interval(min - half, max + half);
        }
        __device__ float size() const {
            return max - min;
        }
};

__device__ interval operator+(const interval &a, float displacement) {
    return interval(a.min + displacement, a.max + displacement);
}

__device__ interval operator+(float displacement, const interval &a) {
    return a + displacement;
}

#endif