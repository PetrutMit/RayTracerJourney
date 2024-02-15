#include "Headers/rtweekend.hpp"
#include "Headers/color.hpp"
#include "Headers/hittable_list.hpp"
#include "Headers/sphere.hpp"
#include "Headers/camera.hpp"
#include "Headers/lambertian.hpp"
#include "Headers/metal.hpp"
#include "Headers/dielectric.hpp"

#include <iostream>
#include <fstream>

#define MASTER 0

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#include <mpi.h>

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

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}



int main(int argc, char** argv) {

    // Image
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 900;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 10;
    const int max_depth = 50;

    // Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;
    camera cam = camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    hittable_list world = random_scene();

    //! Parallel rendering
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int start, end;

    // Each process renders its own part of the image
    start = rank * image_height / size;
    end = MIN((rank + 1) * image_height / size, image_height);

    // Render to framebuffer
    color *framebuffer = new color[image_width * (end - start)];

    for (int i = start; i < end; i ++) {
        for (int j = 0; j < image_width; j ++) {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; s ++) {
                auto u = (j + random_double()) / (image_width-1);
                auto v = (i + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
           framebuffer[(i - start) * image_width + j] = pixel_color;
        }
    }

    // !Gather all the frame buffers to the master process
    color *framebuffer_all;
    if (rank == MASTER) {
        framebuffer_all = new color[image_width * image_height];
    }

    // Define a new MPI data type for the frame buffer
    // color consists of a vec3, which is 3 doubles
    MPI_Datatype MPI_framebuffer;
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_framebuffer);
    MPI_Type_commit(&MPI_framebuffer);

    MPI_Gather(framebuffer, image_width * (end - start), MPI_framebuffer, framebuffer_all, image_width * (end - start), MPI_framebuffer, MASTER, MPI_COMM_WORLD);

    // Master process writes the image to a file
    if (rank == MASTER) {
        std::ofstream outfile;
        outfile.open("image.ppm");
        outfile << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int i = image_height; i >= 0; i --) {
            for (int j = 0; j < image_width; j ++) {
                write_color(outfile, framebuffer_all[i * image_width + j], samples_per_pixel);
            }
        }

        outfile.close();
    }

    MPI_Type_free(&MPI_framebuffer);
    delete[] framebuffer;
    
    if (rank == MASTER) {
        delete[] framebuffer_all;
    }

    MPI_Finalize();
}

