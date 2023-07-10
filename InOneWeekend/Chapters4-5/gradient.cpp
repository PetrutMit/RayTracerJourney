#include "../Chapters2-3/Vec3.hpp"
#include "../Chapters2-3/Color.hpp"
#include "Ray.hpp"

#include <iostream>
#include <fstream>

color ray_color (const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    // Make t to be in range [0, 1]
    auto t = 0.5 * (unit_direction.y() + 1.0);
    // Linear blend (lerp)
    // blendedValue=(1−t)⋅startValue+t⋅endValue
    // Start is white, end is a sort of blue
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color (0.5, 0.7, 1.0);
}


int main(void) {

    // The ouput will be a ppm image
    std::ofstream ppm_file("gradient.ppm");

    // Image
    const auto ascpect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / ascpect_ratio);

    // Camera
    auto viewport_height = 2.0;
    auto viewport_width = ascpect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    // Render

    ppm_file << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; j --) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i ++) {
            auto u = double(i) / (image_width - 1);
            auto v = double(j) / (image_height - 1);
            ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
            color pixel_color = ray_color(r);
            write_color(ppm_file, pixel_color, 1);
        }
    }

    std::cerr << "\nDone.\n";

}