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

__device__ void getColor(color& pixelColor, int samplesPerPixel) {
    float r = pixelColor.x();
    float g = pixelColor.y();
    float b = pixelColor.z();

    // Multiple samples and gamma correction
    float scale = 1.0f / samplesPerPixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    pixelColor = color(r, g, b);

    for (int i = 0; i < 3; i++) {
        pixelColor[i] = clamp(pixelColor[i], 0.0f, 0.999f);
    }

    //pixelColor = 256.0f * pixelColor;
}

#endif