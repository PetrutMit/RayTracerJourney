#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "Chapters4-5/Ray.hpp"

struct hit_record {
    point3 p;
    vec3 normal;
    double t;
    bool front_face;

    inline void set_face_normal(const ray &r, const vec3 &outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};
// Const is used in conjunction with virtual to indicate that the function is pure virtual.
// That means it must be overridden by a sub-class. Classes containing pure virtual functions
// are sometimes described as abstract because they cannot be directly instantiated

class hittable {
    public:
        virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};

#endif