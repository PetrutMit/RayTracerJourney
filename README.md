#### Mitrache Mihnea

# Real Time Ray-Tracer

## Starting the project
> A good starting point is Peter Shirley's book: [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

> Steps taken after reading the book:
> * Chapters 2-3:
> 1. Exploring the simplest image format: PPM -> [PPM_format.cpp](Chapters2-3/PPM_format.cpp)
> 2. Exploring vec3 format to get our vec3 class -> [vec3.hpp](Chapters2-3/vec3.hpp)
> 3. Using vec3 to provide some color functions -> [color.hpp](Chapters2-3/color.hpp)
> 4. Integrating the above concepts in getting the PPM image -> [PPM_image.cpp](Chapters2-3/PPM_image.cpp)

> * Chapters 4-5:
> 1. Exploring the ray class -> [ray.hpp](Chapters4-5/ray.hpp)
> 2. Rendering a blue to white gradient -> [gradient.cpp](Chapters4-5/gradient.cpp)
> 3. Gradient example with intersection of a simple sphere -> [gradient_sphere.cpp](Chapters4-5/gradient_sphere.cpp)

> * Chapters 6-7:
> 1. Exploring surface normals -> [surface_normals.cpp](Chapters6-7/surface_normals.cpp)
> 2. Abstractization for hittable objects -> [hittable.hpp](Chapters6-7/hittable.hpp)
> 3. Making the sphere extend a hittable object -> [sphere.hpp](Chapters6-7/sphere.hpp)
> 4. Adding front/back face determination
> 5. Adding support for multiple hittable objects -> [hittable_list.hpp](Chapters6-7/hittable_list.hpp)
> 6. Adding a unified header file -> [rtweekend.hpp](Chapters6-7/rtweekend.hpp)
> 7. Merging all the above concepts -> [snormals_sphere_ground.cpp](Chapters6-7/snormals_sphere_ground.cpp)
> 8. Adding a random generator
> 9. Adding a camera class -> [camera.hpp](Chapters6-7/camera.hpp)
> 10. Modify color.hpp to support multy samples
> 11. Rendering image again with multiple samples to avoid aliasing-> [snormals_sphere_ground_multisample.cpp](Chapters6-7/snormals_sphere_ground_multisample.cpp)

> * Chapter 8:
> 1. Updating previous header files to support matt random reflection
> 2. Exploring the use of a random ray
> 3. Limiting the number of bounces to avoid stack overflow from recursive calls
> 4. Rendering a diffuse sphere with the above concepts -> [diffuse_sphere.cpp](Chapter8/diffuse_sphere.cpp)
> 5. Adding some gamma correction
> 6. Fixing the shadow acne problem
> 7. Exploring alternative methods for diffuse reflection

> * Chapter 9:
> 1. Adding support for materials -> [material.hpp](Chapter9/material.hpp)
> 2. Modifying the hittable class to support materials
> 3. Modelating diffuse/matt materials -> [lambertian.hpp](Chapter9/lambertian.hpp)
> 4. Adding a metal material + reflexion -> [metal.hpp](Chapter9/metal.hpp)
> 5. Rendering a scene with shiny and matt spheres -> [shiny_matt_spheres.cpp](Chapter9/shiny_matt_spheres.cpp)
> 6. Adding a fuzziness factor to the metal material

> * Chapter 10:
> Modifying the classes to support refraction
> Adding a dielectric, always refracting material -> [dielectric.hpp](Chapter10/dielectric.hpp)
> Rendering a scene with dielectric spheres -> [dielectric_spheres.cpp](Chapter10/dielectric_spheres.cpp)
> Checking for total internal reflection