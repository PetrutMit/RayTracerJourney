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
        hittable **hittable_objects;

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

    // Should also take here an iterative approach instead of
    // the handy recursive one
    const bvh_node *stack[64];
    int stackTop = 0;
    bool hit;

    stack[stackTop++] = this;

    while (stackTop) {
        const bvh_node *currentNode = stack[--stackTop];

        if (!currentNode->box.hit(r, t_min, t_max))
            continue;

        if (currentNode->left == currentNode->right) {
            // Reached a leaf node
            hit = currentNode->left->hit(r, t_min, t_max, rec);
            if (hit)
                return true;
        } else {
            // Push the children onto the stack
            stack[stackTop++] = (bvh_node*)currentNode->left;
            stack[stackTop++] = (bvh_node*)currentNode->right;
        }
    }

    return false;
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
        int nodeSize;
        if (n % size == 0)
            nodeSize = n/size;
        else
            nodeSize = n/size + 1;

        // Sort sublists of objects along axis
        for (int i = 0; i < n; i += nodeSize) {
            //Sort the list of objects along the axis
            int left = i;
            int right = i + nodeSize;
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

#endif
