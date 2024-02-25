#ifndef HIITABLELIST_CUH
#define HIITABLELIST_CUH

#include "hittable.cuh"
#include "aabb.cuh"

// This one uses both aggregation and inheritance
class hittable_list : public hittable {
    public:
        __device__ hittable_list() {}
        __device__ hittable_list(hittable **l, int n) : list(l), list_size(n) {
            _bbox = aabb();

            for (int i = 0; i < n; i ++) {
                _bbox = aabb(_bbox, l[i]->bounding_box());
            }
        }

        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
        __device__ virtual aabb bounding_box() const override {
            return _bbox;
        }
    
    public:
        hittable **list;
        int list_size;
    
    private:
        aabb _bbox;
};

__device__ bool hittable_list::hit(const ray& r, interval ray_t, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;

    // Loop through all objects in the list and find the closest hit
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

//__device__ bool hittable_list::bounding_box() const {
    // if (list_size == 0) return false;

    // aabb temp_box;
    // bool first_box = true;

    // // Loop through all objects in the list and find the bounding box
    // for (int i = 0; i < list_size; i++) {
    //     if (!list[i]->bounding_box(t0, t1, temp_box)) return false;
    //     box = first_box ? temp_box : surrounding_box(box, temp_box);
    //     first_box = false;
    // }

    // return true;

//    return _bbox;
//}

#endif