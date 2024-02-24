#ifndef ROTATE_Y_CUH
#define ROTATE_Y_CUH

#include "hittable.cuh"
#include "vec3.cuh"
#include "ray.cuh"

__device__ inline float degrees_to_radians(float degrees) {
    return degrees * PI / 180.0f;
}

class rotate_y : public hittable {
    public:
        __device__ rotate_y(hittable *p, float angle);
        __device__ ~rotate_y() {
            delete ptr;
        }

        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
        __device__ virtual aabb bounding_box() const override {
            return bbox;
        }

    public:
        hittable *ptr;
        float sin_theta;
        float cos_theta;
        aabb bbox;
};

__device__ rotate_y::rotate_y(hittable *p, float angle) : ptr(p) {
    float radians = degrees_to_radians(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    bbox = p->bounding_box();

    point3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    point3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                float x = i * bbox.x.max + (1 - i) * bbox.x.min;
                float y = j * bbox.y.max + (1 - j) * bbox.y.min;
                float z = k * bbox.z.max + (1 - k) * bbox.z.min;

                float newx = cos_theta * x + sin_theta * z;
                float newz = -sin_theta * x + cos_theta * z;

                vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; ++c) {
                    min[c] = fminf(min[c], tester[c]);
                    max[c] = fmaxf(max[c], tester[c]);
                }
            }
        }
    }

    bbox = aabb(min, max);
}

__device__ bool rotate_y::hit(const ray& r, interval ray_t, hit_record& rec) const {
    point3 origin = r.origin();
    vec3 direction = r.direction();

    origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
    origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

    direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
    direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

    ray rotated_r(origin, direction, r.time());

    if (!ptr->hit(rotated_r, ray_t, rec)) {
        return false;
    }

    point3 p = rec.p;
    vec3 normal = rec.normal;

    p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
    p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

    normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
    normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

    rec.p = p;
    rec.set_face_normal(rotated_r, normal);

    return true;
}


#endif