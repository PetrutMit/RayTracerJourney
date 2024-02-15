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

#include <pthread.h>
#include "Headers/thpool.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

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

typedef struct {
    int thread_id;
    hittable_list *world;
    pthread_spinlock_t *lock;
} thread_world_construction_args;

//! Parallel world construction
void *parallel_world_construction(void *arg) {
    thread_world_construction_args *args = (thread_world_construction_args *)arg;
    int thread_id = args->thread_id;
    hittable_list *world = args->world;
    pthread_spinlock_t *lock = args->lock;

    const int NUM_SPHERES = 22;
    const float OFFSET = -11.0;
    // We have to construct NUM_SPHERES * NUM_THREADS spheres

    int start = thread_id * NUM_SPHERES / NUM_THREADS;
    int end = MIN((thread_id + 1) * NUM_SPHERES / NUM_THREADS, NUM_SPHERES);

    int i, j;
    for (i = start; i < end; i++) {
        for (j = 0; j < 22; j++) {
            auto choose_mat = random_double();
            point3 center(i + 0.9*random_double() + OFFSET, 0.2, j + 0.9*random_double() + OFFSET);

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;
                sphere s;
                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                }
                s = sphere(center, 0.2, sphere_material);
                pthread_spin_lock(lock);
                world->add(make_shared<sphere>(s));
                pthread_spin_unlock(lock);
            }
        }
    }
    pthread_exit(NULL);
}

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    pthread_t threads[NUM_THREADS];
    thread_world_construction_args args[NUM_THREADS];
    pthread_spinlock_t lock;
    pthread_spin_init(&lock, 0);

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].world = &world;
        args[i].lock = &lock;
        pthread_create(&threads[i], NULL, parallel_world_construction, (void *)&args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_spin_destroy(&lock);

    return world;
}

typedef struct {
    int thread_id;
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;
    camera *cam;
    hittable_list *world;
    threadpool *pool;

    color *framebuffer;

} parallel_rendering_args;

typedef struct {
    int image_width;
    int image_height;
    int max_depth;
    camera *cam;
    hittable_list *world;
    int i, j;

    color *framebuffer;
} parallel_sampling_args;

pthread_spinlock_t lock;

void parallel_sampling(void *arg) {
    parallel_sampling_args *args = (parallel_sampling_args *)arg;

    int image_width = args->image_width;
    int image_height = args->image_height;
    int max_depth = args->max_depth;
    camera *cam = args->cam;
    hittable_list *world = args->world;
    int i = args->i;
    int j = args->j;
    
    color *framebuffer = args->framebuffer;

    color pixel_color(0, 0, 0);
    auto u = (j + random_double()) / (image_width-1);
    auto v = (i + random_double()) / (image_height-1);

    ray r = cam->get_ray(u, v);
    pixel_color = ray_color(r, *world, max_depth);

    pthread_spin_lock(&lock);
    framebuffer[i * image_width + j] += pixel_color;
    pthread_spin_unlock(&lock);

    return;
}

//! Parallel rendering and parallel sampling
// ! Imbricated parallelism
void *parallel_rendering(void *arg) {
    parallel_rendering_args *args_render = (parallel_rendering_args *)arg;
    int thread_id = args_render->thread_id;
    int image_width = args_render->image_width;
    int image_height = args_render->image_height;
    int max_depth = args_render->max_depth;
    int samples_per_pixel = args_render->samples_per_pixel;
    camera *cam = args_render->cam;
    hittable_list *world = args_render->world;

    color *framebuffer = args_render->framebuffer;

    int start = thread_id * image_height / NUM_THREADS;
    int end = MIN((thread_id + 1) * image_height / NUM_THREADS, image_height);
    int i, j;

    // Sampling must also be done in parallel
    // Because sampling is done for each pixel, a thread pool is needed
    for (int i = start; i < end; i ++) {
        for (int j = 0; j < image_width; j ++) {
            // For every sample add a job to the thread pool
            for (int s = 0; s < samples_per_pixel; s ++) {
                parallel_sampling_args *args_sample = (parallel_sampling_args *)malloc(sizeof(parallel_sampling_args));
                args_sample->image_width = image_width;
                args_sample->image_height = image_height;
                args_sample->max_depth = max_depth;
                args_sample->cam = cam;
                args_sample->world = world;
                args_sample->i = i;
                args_sample->j = j;
                args_sample->framebuffer = framebuffer;

                thpool_add_work(*args_render->pool, parallel_sampling, (void *)args_sample);
            }
        }
    }
   
    pthread_exit(NULL);
}

int main(int argc, char **argv) {

    if (argc != 2) {
        std::cerr << "Usage: ./final_scene <NUM_THREADS>" << std::endl;
        exit(1);
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

    // !Parallel rendering
    pthread_t threads[NUM_THREADS];
    parallel_rendering_args args[NUM_THREADS];
    color *framebuffer = new color[image_width * image_height];
    threadpool pool = thpool_init(NUM_THREADS);
    int i;

    pthread_spin_init(&lock, 0);

    for (i = 0; i < NUM_THREADS; i ++) {
        args[i].thread_id = i;
        args[i].image_width = image_width;
        args[i].image_height = image_height;
        args[i].samples_per_pixel = samples_per_pixel;
        args[i].max_depth = max_depth;
        args[i].cam = &cam;
        args[i].world = &world;
        args[i].framebuffer = framebuffer;
        args[i].pool = &pool;
    }

    for (i = 0; i < NUM_THREADS; i ++) {
        pthread_create(&threads[i], NULL, parallel_rendering, (void *)&args[i]);
    }

    thpool_wait(pool);
    thpool_destroy(pool);

    for (i = 0; i < NUM_THREADS; i ++) {
        pthread_join(threads[i], NULL);
    }

    // Write to file framebuffer
    std::ofstream ppm_file("final_scene.ppm");
    ppm_file << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int i = image_height - 1; i >= 0; i --) {
        for (int j = 0; j < image_width; j ++) {
            color pixel_color = framebuffer[i * image_width + j];
            write_color(ppm_file, pixel_color, samples_per_pixel);
        }
    }
}
