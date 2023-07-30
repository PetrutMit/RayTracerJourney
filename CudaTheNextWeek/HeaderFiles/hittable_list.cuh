#ifndef HIITABLELIST_CUH
#define HIITABLELIST_CUH

#include "hittable.cuh"
#include "aabb.cuh"


// This one uses both aggregation and inheritance
class hittable_list : public hittable {
    public:
        __device__ hittable_list() {}
        __device__ hittable_list(hittable **l, int n) : list(l), list_size(n) {}
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

        hittable **list;
        int list_size;
};

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    // Loop through all objects in the list and find the closest hit
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__device__ bool hittable_list::bounding_box(float t0, float t1, aabb& box) const {
    if (list_size == 0) return false;

    aabb temp_box;
    bool first_box = true;

    // Loop through all objects in the list and find the bounding box
    for (int i = 0; i < list_size; i++) {
        if (!list[i]->bounding_box(t0, t1, temp_box)) return false;
        box = first_box ? temp_box : surrounding_box(box, temp_box);
        first_box = false;
    }

    return true;
}

#endif