#ifndef COLOR_CUH
#define COLOR_CUH

#include "vec3.cuh"

#include <iostream>

// Adding a clamp method
inline __device__ float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ void getColor(color& pixelColor) {
    auto r = pixelColor.x();
    auto g = pixelColor.y();
    auto b = pixelColor.z();

    for (int i = 0; i < 3; i++) {
        pixelColor[i] = clamp(pixelColor[i], 0.0f, 0.999f);
    }

    pixelColor = 256.0f * pixelColor;
}

#endif