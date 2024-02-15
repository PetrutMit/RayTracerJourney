#include "Headers/rtweekend.hpp"
#include "Headers/color.hpp"
#include "Headers/hittable_list.hpp"
#include "Headers/sphere.hpp"
#include "Headers/camera.hpp"
#include "Headers/lambertian.hpp"
#include "Headers/metal.hpp"
#include "Headers/dielectric.hpp"

#include <omp.h>

#include <iostream>
#include <fstream>

int NUM_THREADS;

color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0,0,0);

    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth-1);
        return color(0,0,0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

// Parallelize world construction using OpenMP
hittable_list random_scene() {
    hittable_list world;

    // World has 22x22 random colored spheres + 4 fixed spheres

    // Fixed spheres
    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material)); 

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    // Random spheres
    hittable_list individual_worlds[NUM_THREADS];
    int thread_id, a, b;

    // !Parallel World Construction
    #pragma omp parallel for num_threads(NUM_THREADS) shared(individual_worlds) private(thread_id, a, b)
    for (a = -11; a < 11; a++) {
        thread_id = omp_get_thread_num();
        for (b = -11; b < 11; b++) {
            auto center = point3(a + 0.9*random_double(), 0.2, b + 0.9*random_double());
            sphere s = sphere::make_random_sphere(random_double(), center);
            
            individual_worlds[thread_id].add(make_shared<sphere>(s));
        }
    }

    // Reconstruct world from thread individual worlds
    for (int i = 0; i < NUM_THREADS; i++) {
        world.addList(individual_worlds[i]);
    }

    return world;
}


int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: ./final_scene <NUM_THREADS>" << std::endl;
        return 1;
    }

    NUM_THREADS = atoi(argv[1]);

    // Image
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 900;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 10;
    const int max_depth = 50;

    // World
    auto world = random_scene();

    // Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

 
    // !Parallel Rendering
    // In order to avoid writing to the same file from multiple threads, we will use a frame
    // buffer to store our pixel colors. This is a storage space for the colors of each pixel.
    // When the rendering is done, we will write the frame buffer to the file. 

    std::vector<color> frame_buffer(image_width * image_height);

    int i,j,s;
    int pixel_index;

    // We need nested parallelism here which must be enabled explicitly
    omp_set_nested(1);

    #pragma omp parallel for num_threads(NUM_THREADS) shared(frame_buffer) private(i, j, s, pixel_index)
    for (j = image_height - 1; j >= 0; j --) {
        for (i = 0; i < image_width; i ++) {
            pixel_index = j * image_width + i;
            double color_R = 0, color_G = 0, color_B = 0;
            // More samples for anti-aliasing
            #pragma omp parallel for reduction(+:color_R, color_G, color_B) num_threads(NUM_THREADS) shared(frame_buffer) private(s) 
            for (s = 0; s < samples_per_pixel; s ++) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                color pixel_color = ray_color(r, world, max_depth);
                color_R += pixel_color.x();
                color_G += pixel_color.y();
                color_B += pixel_color.z();
            }
            frame_buffer[pixel_index] = color(color_R, color_G, color_B);
        }
    }

    // Write frame buffer to file
    std::ofstream ppm_file("final_scene.ppm");
    ppm_file << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (j = image_height - 1; j >= 0; j --) {
        for (i = 0; i < image_width; i ++) {
            pixel_index = j * image_width + i;
            write_color(ppm_file, frame_buffer[pixel_index], samples_per_pixel);
        }
    }
}

