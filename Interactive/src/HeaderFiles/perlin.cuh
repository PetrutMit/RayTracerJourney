#ifndef PERLIN_CUH
#define PERLIN_CUH

#include "random.cuh"

#include <math.h>

class perlin {

    public:
        __device__ perlin() {}
        __device__ perlin(curandState *localRandState) {
            ranvec = new vec3[perlin::point_count];
            for (int i = 0; i < perlin::point_count; ++i) {
                ranvec[i] = unit_vector(randomVectorBetween(localRandState, -1, 1));
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
            float u = p.x() - floor(p.x());
            float v = p.y() - floor(p.y());
            float w = p.z() - floor(p.z());

            int i = floor(p.x());
            int j = floor(p.y());
            int k = floor(p.z());

            vec3 c[2][2][2];

            for (int di = 0; di < 2; di ++) {
                for (int dj = 0; dj < 2; dj ++) {
                    for (int dk = 0; dk < 2; dk ++) {
                        c[di][dj][dk] = ranvec[
                            perm_x[(i + di) & 255] ^
                            perm_y[(j + dj) & 255] ^
                            perm_z[(k + dk) & 255]
                        ];
                    }
                }
            }

            return perlin_interp(c, u, v, w);
        }

        __device__ float turb(const point3& p, int depth = 7) const {
            float accum = 0.0f;
            point3 temp_p = p;
            float weight = 1.0f;

            for (int i = 0; i < depth; ++i) {
                accum += weight * noise(temp_p);
                weight *= 0.5f;
                temp_p *= 2;
            }

            return fabs(accum);
        }

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
        __device__ static float perlin_interp(vec3 c[2][2][2], float u, float v, float w) {
            // Hermitian smoothing
            float uu = u * u * (3 - 2 * u);
            float vv = v * v * (3 - 2 * v);
            float ww = w * w * (3 - 2 * w);

            float accum = 0.0;

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    for (int k = 0; k < 2; ++k) {
                        vec3 weight_v(u - i, v - j, w - k);
                        accum += (i * uu + (1 - i) * (1 - uu)) *
                                 (j * vv + (1 - j) * (1 - vv)) *
                                 (k * ww + (1 - k) * (1 - ww)) *
                                 dot(c[i][j][k], weight_v);
                    }
                }
            }
            return accum;
        }

        static const int point_count = 256;
        vec3 *ranvec;
        int *perm_x;
        int *perm_y;
        int *perm_z;
};

#endif
