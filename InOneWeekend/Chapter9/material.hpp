#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "../Chapters6-7/rtweekend.hpp"

struct hit_record;

class material {
    public:
        virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};


#endif