#ifndef MOVING_SPHERE_CUH
#define MOVING_SPHERE_CUH

#include "hittable.cuh"
#include "vec3.cuh"
#include "material.cuh"
#include "aabb.cuh"

#define PI 3.141592653589793f

class sphere : public hittable {
    public:
        __device__ sphere() {}
        
        // Call to modelate a static sphere
        __device__ sphere(point3 cen0, float r, material *m) : _center0(cen0), _radius(r), _mat_ptr(m), _is_moving(false) {
            vec3 radius3(r, r, r);
            _bbox = aabb(_center0 - radius3, _center0 + radius3);
        }

        // Call to modelate a moving sphere -> Should be removed
        __device__ sphere(point3 cen0, point3 cen1, float r, material *m): _center0(cen0), _radius(r), _mat_ptr(m), _is_moving(true) {
            vec3 radius3(r, r, r);
            aabb box0(cen0 - radius3, cen0 + radius3);
            aabb box1(cen1 - radius3, cen1 + radius3);
            _bbox = aabb(box0, box1);
            _direction = cen1 - cen0;
        };

        __device__ aabb bounding_box() const override {
            return _bbox;
        }
        
        __device__  bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
        __device__ point3 center(float time) const;

    private:
        __device__ static void get_sphere_uv(const vec3& p, float& u, float& v) {
            float theta = acos(-p.y());
            float phi = atan2(-p.z(), p.x()) + PI;

            u = phi / (2*PI);
            v = theta / PI;
        }
    
    private:
        point3 _center0;
        float _radius;
        material *_mat_ptr;
        bool _is_moving;
        vec3 _direction;
        aabb _bbox;
};

__device__ point3 sphere::center(float tm) const {
    if (!_is_moving) return _center0;
    return _center0 + tm * _direction;
}

__device__ bool sphere::hit(const ray& r, interval ray_t, hit_record& rec) const {
    vec3 oc = r.origin() - center(r.time());
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - _radius*_radius;

    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (-half_b + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center(r.time())) / _radius;
    rec.set_face_normal(r, outward_normal);
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = _mat_ptr;

    return true;
}


#endif