#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hittable.hpp"
#include "Vec3.hpp"
#include "lambertian.hpp"
#include "metal.hpp"
#include "dielectric.hpp"

class sphere : public hittable {
    public:
        sphere() {}
        sphere(point3 cen, double r, shared_ptr<material> m) : center(cen), radius(r), mat_ptr(m) {};
        
        static sphere make_random_sphere(double random, point3 center);
        virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;

    public:
        point3 center;
        double radius;
        shared_ptr<material> mat_ptr;
};

bool sphere::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    vec3 oc = r.origin() - center;
    // Solve the equation
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = half_b*half_b - a*c;

    // Consider just one root
    if (discriminant < 0) {
        return false;
    } else {
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range
        auto root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root) {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        // Determine the face normal
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;

        return true;
    }
}

sphere sphere::make_random_sphere(double random, point3 center) {
    // Choose a random material
    shared_ptr<material> material_ptr;
    auto choose_mat = random_double();
    if (choose_mat < 0.8) {
        // Diffuse
        auto albedo = color::random() * color::random();
        material_ptr = make_shared<lambertian>(albedo);
    } else if (choose_mat < 0.95) {
        // Metal
        auto albedo = color::random(0.5, 1);
        auto fuzz = random_double(0, 0.5);
        material_ptr = make_shared<metal>(albedo, fuzz);
    } else {
        // Glass
        material_ptr = make_shared<dielectric>(1.5);
    }

    // Choose a random radius
    auto radius = random_double(0.2, 0.5);

    return sphere(center, radius, material_ptr);
}

#endif