#include "Chapters2-3/vec3.hpp"
#include "Chapters2-3/Color.hpp"
#include "Chapters4-5/Ray.hpp"

#include <iostream>
#include <fstream>


double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    // Solve the equation
    auto a = dot(r.direction(), r.direction());
    auto half_b = dot(oc, r.direction());
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = half_b*half_b - 4*a*c;

    // Consider just one root
    if (discriminant < 0) {
        return -1.0;
    } else {
        return (-half_b - sqrt(discriminant)) / a;
    }
}

color ray_color(const ray& r) {
    auto t = hit_sphere(point3(0,0,-1), 0.5, r);

    if (t > 0.0) {
        // Normal vector P - C
        vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
        // Map the normal vector to the color
        return 0.5 * color(N.x()+1, N.y()+1, N.z()+1);
    }
    vec3 unit_direction = unit_vector(r.direction());
    t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}


int main(void) {

    // The ouput will be a ppm image
    std::ofstream ppm_file("gradient_sphere_normal.ppm");

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
            write_color(ppm_file, pixel_color);
        }
    }

    std::cerr << "\nDone.\n";
}