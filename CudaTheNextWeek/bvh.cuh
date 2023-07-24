#ifndef BVH_H
#define BVH_H


#include "hittable.cuh"
#include "hittable_list.cuh"

#include "ray.h"
#include <curand_kernel.h>

#include <thrust/device_vector.h>

class bvh_node : public hittable {
    public:
        __device__ bvh_node() {}
        __device__ bvh_node(hittable_list *l, float time0, float time1, curandState *localState) : bvh_node(l->list, 0, l->size, time0, time1, localState) {}
        __device__ bvh_node(hittable **l, int n, float time0, float time1, curandState *localState);
        __device__ bvh_node(hittable *left, hittable *right, aabb box)
            : left(left), right(right), box(box) {}

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

    public:
        hittable *left;
        hittable *right;
        aabb box;
};

__device__  bool box_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis) {
    aabb box_a;
    aabb box_b;

    if (!a->bounding_box(0,0, box_a) || !b->bounding_box(0,0, box_b))
        std::cerr << "No bounding box in bvh_node constructor.\n";

    return box_a.min().e[axis] < box_b.min().e[axis];
}


__device__ bool box_x_compare (const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
    return box_compare(a, b, 0);
}

__device__ bool box_y_compare (const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
    return box_compare(a, b, 1);
}

__device__ bool box_z_compare (const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
    return box_compare(a, b, 2);
}


__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const {
    b = box;
    return true;
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!box.hit(r, t_min, t_max))
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

__device__ bvh_node::bvh_node(hittable **l, int n, float time0, float time1, curandState *localState) {


    if (n == 1) {
        left = right = l[0];
    }

    //Bottom up approach to build tree
    for (int size = 1; size < n; size *= 2) {
        //Choose random axis to split on
        int axis = ceilf(3 * curand_uniform(localState));
        //Choose comparator function based on axis
        void *comparator;
        switch (axis) {
            case 0:
                comparator = box_x_compare;
                break;
            case 1:
                comparator = box_y_compare;
                break;
            case 2:
                comparator = box_z_compare;
                break;
        }
        // Sort sublists of objects along axis
        for (int i = 0; i < n; i += n/size) {
            //Sort the list of objects along the axis
            thrust::sort(thrust::device, l + i, l + i + n/size, comparator);
        }
    }

    //Build tree from sorted list
    //Bottom up approach to build tree

    //Leaf node
    for (int i = 0; i < n; i ++) {
        l[i] = new bvh_node(l[i], l[i], time0, time1, localState);
    }

    for (int size = 1; size < n; size *= 2) {
        for (int i = 0; i < n; i += size * 2) {
            //Combine two nodes into one
            l[i] = new bvh_node(l[i], l[i + size * 2], time0, time1, localState);
        }
    }
}

#endif
