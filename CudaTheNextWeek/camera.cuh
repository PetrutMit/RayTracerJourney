#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "ray.cuh"
#include <curand_kernel.h>
#define PI 3.141592653589793f

__device__ vec3 randomInUnitDisk(curandState *state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(state), curand_uniform(state), 0.0f) - vec3(1.0f, 1.0f, 0.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

__device__ float randomFloat(curandState *state, float min, float max) {
    return min + (max - min) * curand_uniform(state);
}

class camera {
    public:
        __device__ camera(
            point3 lookfrom,
            point3 lookat,
            vec3 vup,
            float vfov,
            float aspect_ratio,
            float aperture,
            float focus_dist,
            float _time0 = 0.0f,
            float _time1 = 0.0f
        ) {
            float theta = degrees_to_radians(vfov);
            float h = tan(theta / 2.0f);
            float viewport_height = 2.0f * h;
            float viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - focus_dist * w;

            lens_radius = aperture / 2.0f;
            time0 = _time0;
            time1 = _time1;
        }

        __device__ ray get_ray(float s, float t, curandState *state) const {
            vec3 rd = lens_radius * randomInUnitDisk(state);
            vec3 offset = u * rd.x() + v * rd.y();

            return ray(
                origin + offset,
                lower_left_corner + s * horizontal + t * vertical - origin - offset,
                randomFloat(state, time0, time1)
            );
        }

    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u, v, w;
        float lens_radius;
        float time0, time1; // shutter open/close times

    __device__ static float degrees_to_radians(float degrees) {
        return degrees * PI / 180.0f;
    }
};

#endif

