#### Mitrache Mihnea

# Real Time Ray-Tracer

## Starting the project
> A good starting point is Peter Shirley's book: [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

> Steps taken after reading the book:
> * Chapters 2-3:
> 1. Exploring the simplest image format: PPM -> [PPM_format.cpp](InOneWeekend/Chapters2-3/PPM_format.cpp)
> 2. Exploring vec3 format to get our vec3 class -> [vec3.hpp](InOneWeekend/Chapters2-3/vec3.hpp)
> 3. Using vec3 to provide some color functions -> [color.hpp](InOneWeekend/Chapters2-3/color.hpp)
> 4. Integrating the above concepts in getting the PPM image -> [PPM_image.cpp](InOneWeekend/Chapters2-3/PPM_image.cpp)

> * Chapters 4-5:
> 1. Exploring the ray class -> [ray.hpp](InOneWeekend/Chapters4-5/ray.hpp)
> 2. Rendering a blue to white gradient -> [gradient.cpp](InOneWeekend/Chapters4-5/gradient.cpp)
> 3. Gradient example with intersection of a simple sphere -> [gradient_sphere.cpp](InOneWeekend/Chapters4-5/gradient_sphere.cpp)

> * Chapters 6-7:
> 1. Exploring surface normals -> [surface_normals.cpp](InOneWeekend/Chapters6-7/surface_normals.cpp)
> 2. Abstractization for hittable objects -> [hittable.hpp](InOneWeekend/Chapters6-7/hittable.hpp)
> 3. Making the sphere extend a hittable object -> [sphere.hpp](InOneWeekend/Chapters6-7/sphere.hpp)
> 4. Adding front/back face determination
> 5. Adding support for multiple hittable objects -> [hittable_list.hpp](InOneWeekend/Chapters6-7/hittable_list.hpp)
> 6. Adding a unified header file -> [rtweekend.hpp](InOneWeekend/Chapters6-7/rtweekend.hpp)
> 7. Merging all the above concepts -> [snormals_sphere_ground.cpp](InOneWeekend/Chapters6-7/snormals_sphere_ground.cpp)
> 8. Adding a random generator
> 9. Adding a camera class -> [camera.hpp](InOneWeekend/Chapters6-7/camera.hpp)
> 10. Modify color.hpp to support multy samples
> 11. Rendering image again with multiple samples to avoid aliasing-> [snormals_sphere_ground_multisample.cpp](InOneWeekend/Chapters6-7/snormals_sphere_ground_multisample.cpp)

> * Chapter 8:
> 1. Updating previous header files to support matt random reflection
> 2. Exploring the use of a random ray
> 3. Limiting the number of bounces to avoid stack overflow from recursive calls
> 4. Rendering a diffuse sphere with the above concepts -> [diffuse_sphere.cpp](InOneWeekend/Chapter8/diffuse_sphere.cpp)
> 5. Adding some gamma correction
> 6. Fixing the shadow acne problem
> 7. Exploring alternative methods for diffuse reflection

> * Chapter 9:
> 1. Adding support for materials -> [material.hpp](InOneWeekend/Chapter9/material.hpp)
> 2. Modifying the hittable class to support materials
> 3. Modelating diffuse/matt materials -> [lambertian.hpp](InOneWeekend/Chapter9/lambertian.hpp)
> 4. Adding a metal material + reflexion -> [metal.hpp](InOneWeekend/Chapter9/metal.hpp)
> 5. Rendering a scene with shiny and matt spheres -> [shiny_matt_spheres.cpp](InOneWeekend/Chapter9/shiny_matt_spheres.cpp)
> 6. Adding a fuzziness factor to the metal material

> * Chapter 10:
> 1. Modifying the classes to support refraction
> 2. Adding a dielectric, always refracting material -> [dielectric.hpp](InOneWeekend/Chapter10/dielectric.hpp)
> 3. Rendering a scene with dielectric spheres -> [dielectric_spheres.cpp](InOneWeekend/Chapter10/dielectric_spheres.cpp)
> 4. Checking for total internal reflection
> 5. Integrating Schlick's approximation for reflectivity
> 6. Negative radius for hollow glass spheres with dielectric material -> [hollow_glass_spheres.cpp](InOneWeekend/Chapter10/hollow_glass_spheres.cpp)

> * Chapter 11:
> 1. Adding an ajustable field of view to our camera
> 2. Rending a scene and using our adjustable camera -> [fov_camera.cpp](InOneWeekend/Chapter11/vfov_camera.cpp)
> 3. Actualizing camera to have a lookat, lookfrom and vup vectors
> 4. Applying the above concepts to get different viewpoints -> [viewpoints.cpp](InOneWeekend/Chapter11/viewpoints.cpp)

> * Chapters 12-13:
> 1. Introducing the focus concept on our camera
> 2. More infos on camera dependencies -> [VideoAperture](InOneWeekend/https://www.youtube.com/watch?v=YojL7UQTVhc)
> 3. Rendering a scene with depth of field -> [depth_of_field.cpp](InOneWeekend/Chapters12-13/depth_of_field.cpp)
> 4. Summing all the above chapters we can get our beautiful
looking scene -> [final_scene.cpp](InOneWeekend/Chapters12-13/final_scene.cpp)

<hr><hr>

> After understending the concept, it is time to speed up the process using **CUDA**

> Steps taken to make our CUDA implementation:
> 1. Creating a cuda header file for our cuda implement -> [header.cuh](Cuda/header.cuh)
> 2. Adding a cuda color header file -> [color.cuh](Cuda/color.cuh)
> 3. Adding a cuda vec3 header file -> [vec3.cuh](Cuda/vec3.cuh)
> 4. Rendering our first image using CUDA -> [firstCuda.cu](Cuda/firstCuda.cu)
> 5. Adding a cuda ray header file -> [ray.cuh](Cuda/ray.cuh)
> 6. Rendering a gradient image using rays and a simple camera -> [gradient.cu](Cuda/gradient.cu)
> 7. Introducing a simple sphere -> [simpleSphere.cu](Cuda/simpleSphere.cu)
> 8. Rendering surface normals on a sphere -> [surfaceNormals.cu](Cuda/surfaceNormals.cu)
> 9. Adding a cuda sphere header file -> [sphere.cuh](Cuda/sphere.cuh)
> 10. Adding a cuda hittable header file -> [hittable.cuh](Cuda/hittable.cuh)
> 11. Adding a cuda hittable list header file -> [hittableList.cuh](Cuda/hittableList.cuh)
> 12. Rendering a scene with multiple hittable objects -> [multipleSpheres.cu](Cuda/multipleSpheres.cu)
> 13. Tested ways of allocating hittable objects on the device -> [testAllocation.cu](Cuda/testAllocation.cu)
> 14. Adding a cuda camera header file -> [camera.cuh](Cuda/camera.cuh)
> 15. Adding random support to get anti-aliasing in cuda -> [multipleSpheresAA.cu](Cuda/multipleSpheresAA.cu)
> 16. Transforming ray bouncing recursive calls into iterative calls
> 17. Rendering a scene with diffuse materials -> [diffuseSpheres.cu](Cuda/diffuseSpheres.cu)
> 18. Adding a cuda material header file -> [material.cuh](Cuda/material.cuh)
> 19. Adding material support to our hittable objects
> 20. Adding support for lambertian material -> [material.cuh](Cuda/material.cuh)
> 21. Adding support for metal material -> [material.cuh](Cuda/material.cuh)
> 22. Rendering metal spheres -> [metalSpheres.cu](Cuda/metalSpheres.cu)
> 23. Adding support for dielectric material -> [material.cuh](Cuda/material.cuh)
> 24. Rendering dielectric spheres -> [dielectricSpheres.cu](Cuda/dielectricSpheres.cu)
> 25. Adding a cuda camera header file -> [camera.cuh](Cuda/camera.cuh)
> 26. Exploring various camera positions -> [viewpoints.cu](Cuda/viewpoints.cu)