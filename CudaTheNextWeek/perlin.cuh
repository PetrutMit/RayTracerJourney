#ifndef PERLIN_CUH
#define PERLIN_CUH

#include "random.cuh"

class perlin {

    public:
        __device__ perlin() {}
        __device__ perlin(curandState *localRandState) {
            ranvec = new float[perlin::point_count];
            for (int i = 0; i < perlin::point_count; ++i) {
                ranvec[i] = randomFloat(localRandState, 0, 1);
            }
            perm_x = perlin_generate_perm(localRandState);
            perm_y = perlin_generate_perm(localRandState);
            perm_z = perlin_generate_perm(localRandState);
        }

        __device__ ~perlin() {
            delete[] ranvec;
            delete[] perm_x;
            delete[] perm_y;
            delete[] perm_z;
        }

        __device__ float noise(const point3& p) const {
            int i = static_cast<int>(4*p.x()) & 255;
            int j = static_cast<int>(4*p.y()) & 255;
            int k = static_cast<int>(4*p.z()) & 255;

            int contribution = perm_x[i] ^ perm_y[j] ^ perm_z[k];
            return ranvec[contribution];
        }

    private:
        __device__ static int* perlin_generate_perm(curandState *localRandState) {
            int *p = new int[perlin::point_count];

            for (int i = 0; i < perlin::point_count; ++i) {
                p[i] = i;
            }

            for (int i = perlin::point_count - 1; i > 0; --i) {
                int target = randomInt(localRandState, 0, i);
                int tmp = p[i];
                p[i] = p[target];
                p[target] = tmp;
            }

            return p;
        }

    private:
        static const int point_count = 256;
        float *ranvec;
        int *perm_x;
        int *perm_y;
        int *perm_z;
};

#endif
