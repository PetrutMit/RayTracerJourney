/* Testing dynamic memory allocation on the GPU
 * Test whether we get better performace if we allocate dynamic memory via new
  on the GPU or via cudaMalloc
 * We need to allocate a vector of pointers to int
*/
#include <iostream>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// GPU allocation manner
__global__ void allocateOnGPU(int **ptr, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < size; i++) {
            ptr[i] = new int;
        }
    }
}

__global__ void deallocateOnGPU(int **ptr, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < size; i++) {
            delete ptr[i];
        }
    }
}

// CPU allocation manner
__host__ void allocateViaCPU(int **ptr, int size) {
    // This segfaults because even though ptr is allocated in the GPU, to use cudaMalloc
    // you need to dereference it in the CPU.
    for (int i = 0; i < size; i ++) {
        checkCudaErrors(cudaMalloc((void **)&ptr[i], sizeof(int)));
    }
}

__host__ void deallocateViaCPU(int **ptr, int size) {
    for (int i = 0; i < size; i ++) {
        checkCudaErrors(cudaFree(ptr[i]));
    }
}

__global__ void doSomeWork(int **ptr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        *ptr[idx] = 1;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <true|false>" << std::endl;
        return 1;
    }
    bool sw = atoi(argv[1]) == 1 ? true : false;

    int  **ptrDev;
    const int size = 1e4;

    checkCudaErrors(cudaMalloc((void **)&ptrDev, size * sizeof(int*)));

   if (sw == true) {
        allocateOnGPU<<<1, 1>>>(ptrDev, size);
        checkCudaErrors(cudaDeviceSynchronize());

        dim3 blockSize(1024);
        dim3 blockCount((size + blockSize.x - 1) / blockSize.x);
        doSomeWork<<<blockCount, blockSize>>>(ptrDev, size);
        checkCudaErrors(cudaDeviceSynchronize());

        deallocateOnGPU<<<1, 1>>>(ptrDev, size);
        checkCudaErrors(cudaDeviceSynchronize());
    } else {
        // CPU allocation
        allocateViaCPU(ptrDev, size);

        dim3 blockSize(1024);
        dim3 blockCount((size + blockSize.x - 1) / blockSize.x);
        doSomeWork<<<blockCount, blockSize>>>(ptrDev, size);
        checkCudaErrors(cudaDeviceSynchronize());

        deallocateViaCPU(ptrDev, size);
   }

    checkCudaErrors(cudaFree(ptrDev));

    return 0;
}
