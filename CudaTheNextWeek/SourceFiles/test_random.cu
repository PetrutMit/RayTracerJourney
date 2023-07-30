#include "random.cuh"

#define SIZE 1e3

#define THREADS_PER_BLOCK 256

// Adding a method to help us handle errors
inline void checkReturn(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__global__ void randomInit(curandState *state) {
    if (threadIdx.x == 0) {
        curand_init(1234, 0, 0, state);
    }
}

__global__ void makeRandoms(float *vec_float, int *vec_int, curandState *state) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < SIZE; i ++) {
            vec_float[i] = randomFloat(state, 0, 1);
            vec_int[i] = randomInt(state, 0, 256);
        }
    }
}

int main(void) {
    float *gpu_vec_float;
    checkReturn(cudaMalloc((void **)&gpu_vec_float, SIZE * sizeof(float)));

    int *gpu_vec_int;
    checkReturn(cudaMalloc((void **)&gpu_vec_int, SIZE * sizeof(int)));

    curandState *randomState;
    checkReturn(cudaMalloc((void **)&randomState, sizeof(curandState)));

    randomInit<<<1, 1>>>(randomState);

    makeRandoms<<<1, 1>>>(gpu_vec_float, gpu_vec_int, randomState);

    float *vec_float = (float *)malloc(SIZE * sizeof(float));
    checkReturn(cudaMemcpy(vec_float, gpu_vec_float, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    int *vec_int = (int *)malloc(SIZE * sizeof(int));
    checkReturn(cudaMemcpy(vec_int, gpu_vec_int, SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < SIZE; i ++) {
        std::cout << vec_float[i] << " " <<  vec_int[i] << std::endl;
    }

}