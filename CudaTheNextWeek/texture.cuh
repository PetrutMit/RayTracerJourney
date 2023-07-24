#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include "vec3.cuh"

class my_texture {
    public:
        __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public my_texture {
    public:
        __device__ constant_texture() {}
        __device__ constant_texture(vec3 c) : color_value(c) {}

        __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
            return color_value;
        }

    private:
        vec3 color_value;
};

class checker_texture : public my_texture {
    public:
        __device__ checker_texture() {}
        __device__ checker_texture(my_texture* t0, my_texture* t1) : even(t0), odd(t1) {}

        __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
            float sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
            if (sines < 0) {
                return odd->value(u, v, p);
            } else {
                return even->value(u, v, p);
            }
        }

    public:
        my_texture* odd;
        my_texture* even;
};

#endif