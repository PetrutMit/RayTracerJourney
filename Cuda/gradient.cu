/* Main file to render a white-blue gradient
 * We introduce rays to color pixel along the gradient
*/

#include "header.cuh"

__device__ color rayColor(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*color(1.0f,1.0f,1.0f) + t*color(0.5f,0.7f,1.0f);
}

__global__ void render(color *fb, int maxX, int maxY, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= maxX) || (j >= maxY)) return;
    int pixel_index = j*maxX + i;
    float u = float(i) / float(maxX);
    float v = float(j) / float(maxY);
    ray r(origin, lowerLeftCorner + u*horizontal + v*vertical - origin);
    color pixel_color = rayColor(r);
    getColor(pixel_color);

    fb[pixel_index] = pixel_color;
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

    auto aspect_ratio = float(nx) / float(ny);
    // Camera
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

    render<<<blockCount, blockSize >>>(fb_gpu, nx, ny, lower_left_corner, horizontal, vertical, origin);
    checkReturn(cudaGetLastError());
    checkReturn(cudaDeviceSynchronize());

    // Copy FB from GPU to CPU
    color *fb_cpu = new color[num_pixels];
    cudaStatus = cudaMemcpy(fb_cpu, fb_gpu, num_pixels * sizeof(color), cudaMemcpyDeviceToHost);

    // Output FB as Image
    std::ofstream ppmFile("gradient.ppm");

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


