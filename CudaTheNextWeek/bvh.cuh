#ifndef BVH_H
#define BVH_H


#include "hittable.cuh"
#include "hittable_list.cuh"

#include "ray.cuh"
#include "aabb.cuh"
#include "random.cuh"

#include <thrust/sort.h>

class bvh_node : public hittable {
    public:
        __device__ bvh_node() {}
        __device__ bvh_node(hittable_list *l, float time0, float time1, curandState *localState) : bvh_node(l->list, l->list_size, time0, time1, localState) {}
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

__device__  bool box_compare(hittable *a,hittable *b, int axis) {
    aabb box_a;
    aabb box_b;

    if (!a->bounding_box(0,0, box_a) || !b->bounding_box(0,0, box_b))
        printf("No bounding box in bvh_node constructor.\n");

    return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ bool box_x_compare(hittable *a, hittable *b) {
    return box_compare(a, b, 0);
}

__device__ bool box_y_compare(hittable *a, hittable *b) {
    return box_compare(a, b, 1);
}

__device__ bool box_z_compare(hittable *a, hittable *b) {
    return box_compare(a, b, 2);
}

__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const {
    b = box;
    return true;
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

    if (!box.hit(r, t_min, t_max))
        return false;

    // // Should also take here an iterative approach instead of
    // // the handy recursive one

    // bool found = false;
    // aabb box_left, box_right;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

__device__ bvh_node::bvh_node(hittable **l, int n, float time0, float time1, curandState *localState) {
    if (n == 1) {
        left = right = l[0];
        aabb myBox;
        if (!l[0]->bounding_box(time0, time1, myBox))
            printf("No bounding box in bvh_node constructor.\n");
        box = myBox;
    } else {

    // Get a modifiable copy of the list
    hittable **list = new hittable*[n];
    for (int i = 0; i < n; i++) {
        list[i] = l[i];
    }

    //Bottom up approach to build tree
    for (int size = 1; size < n; size *= 2) {
        //Choose random axis to split on
        int axis = randomInt(localState, 0, 2);
        //Choose comparator function based on axis
        bool (*comparator)(hittable*, hittable*);
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
            int left = i;
            int right = i + n/size;
            if (right > n)
                right = n;
            thrust::sort(list + left, list + right, comparator);
        }
   }

    //Build tree from sorted list
    //Bottom up approach to build tree
    //Leaf node
    bvh_node **nodes = new bvh_node*[n];
    for (int i = 0; i < n; i ++) {
        aabb myBox;
        if (!list[i]->bounding_box(time0, time1, myBox))
            printf("No bounding box in bvh_node constructor.\n");
        nodes[i] = new bvh_node(l[i], l[i], myBox);
    }

    delete[] list;

    // Now we have the leaf nodes, we can build the tree
    while (n > 1) {
        int newSize;
        if (n % 2 == 1)
            newSize = n/2 + 1;
        else
            newSize = n/2;

        int j = 0;
        for (int i = 0; i < newSize; i ++) {
            aabb myBox;
            myBox = surrounding_box(nodes[j]->box, nodes[j + 1]->box);
            nodes[i] = new bvh_node(nodes[j], nodes[j + 1], myBox);
            j += 2;
        }

        n = newSize;
    }

    left = nodes[0];
    right = nodes[1];
    box = surrounding_box(nodes[0]->box, nodes[1]->box);
    }
}

// __device__ bvh_node::bvh_node(hittable **l, int n, float time0, float time1, curandState *state) : hittable_objects(l){

//     // Create a modifiable array of the source scene objects
//     hittable **src_objects = new hittable*[n];

//     for (int i = 0; i < n; i++) {
//         src_objects[i] = hittable_objects[i];
//     }

//     int axis = randomInt(state, 0, 2);
//     bool (*comparator)(hittable*, hittable*);

//     switch (axis) {
//         case 0:
//             comparator = box_x_compare;
//             break;
//         case 1:
//             comparator = box_y_compare;
//             break;
//         case 2:
//             comparator = box_z_compare;
//             break;
//     }

//     if (n == 1) {
//         left = right = src_objects[0];
//     } else if (n == 2) {
//         if (comparator(src_objects[0], src_objects[1])) {
//             left = src_objects[0];
//             right = src_objects[1];
//         } else {
//             left = src_objects[1];
//             right = src_objects[0];
//         }
//     } else {
//         thrust::sort(src_objects, src_objects + n, comparator);

//         int mid = n/2;
//         left = new bvh_node(src_objects, mid, time0, time1, state);
//         right = new bvh_node(src_objects + mid, n - mid, time0, time1, state);
//     }

//     aabb box_left, box_right;

//     if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right))
//         printf("No bounding box in bvh_node constructor.\n");

//     box = surrounding_box(box_left, box_right);
// }


#endif
