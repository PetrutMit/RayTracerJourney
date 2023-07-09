/* Main file to render our first Ray Traced CUDA image
 * The kernel will compute the color of each pixel based
 * on pixel index. So far there are no rays implied
*/

#include "Header.cuh"

__global__ void render(color *fb, int maxX, int maxY) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= maxX) || (j >= maxY)) return;

    // Compute index, color, clamp, and write to FB
    int pixelIndex = j * maxX + i;
    color pixelColor(float(i) / float(maxX), float(j) / float(maxY), 0.2);
    getColor(pixelColor);

    fb[pixelIndex] = pixelColor;

}

int main(void) {
    int nx = 200;
    int ny = 100;

    int num_pixels = nx * ny;

    color *fb_gpu;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&fb_gpu, num_pixels * sizeof(color));
    checkReturn(cudaStatus);

    dim3 blockCount(nx / TX + 1, ny / TY + 1);
    dim3 blockSize(TX, TY);

    render <<<blockCount, blockSize >>> (fb_gpu, nx, ny);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    // Copy FB from GPU to CPU
    color *fb_cpu = new color[num_pixels];
    cudaStatus = cudaMemcpy(fb_cpu, fb_gpu, num_pixels * sizeof(color), cudaMemcpyDeviceToHost);

    // Output FB as Image
    std::ofstream ppmFile("start_image.ppm");

    ppmFile << "P3\n" << nx << " " << ny << "\n255\n";

    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = static_cast<int>(fb_cpu[pixel_index].e[0]);
            int ig = static_cast<int>(fb_cpu[pixel_index].e[1]);
            int ib = static_cast<int>(fb_cpu[pixel_index].e[2]);
            ppmFile << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkReturn(cudaFree(fb_gpu));

}