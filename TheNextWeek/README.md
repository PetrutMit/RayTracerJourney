#### Mitrache Mihnea

## Exploring second book in Peter Shirley's series: Ray Tracing, The Next Week

> Steps taken after reading the book:

> * Chapter 2:
> 1. Adding time stamps to the known classes to get motion
> 2. Adding a moving sphere class -> [moving_sphere.hpp](Chapter2/moving_sphere.hpp)
> 3. Rendering previous final scene with motion -> [moving_spheres.cpp](Chapter2/moving_spheres.cpp)

> * Chapter 3:
> 1. Adding an AABB class -> [aabb.hpp](Chapter3/aabb.hpp)
> 2. Defining bounding boxes for hittable objects
> 3. Adding a BVH class -> [bvh.hpp](Chapter3/bvh.hpp)

> * Chapter 4:
> 1. Adding a texture class -> [texture.hpp](Chapter4/texture.hpp)
> 2. Modelate colors as constant textures
> 3. Adding texture coordinates to the sphere class
> 4. Modify material to support textures
> 5. Adding a checker texture(chess table like) class -> [checker_texture.hpp](Chapter4/texture.hpp)
> 6. Rendering a scene with textures -> [texture_spheres.cpp](Chapter4/texture_spheres.cpp)

> * Chapter 5:
> 1. Adding a perlin noise class -> [perlin.hpp](Chapter5/perlin.hpp)
> 2. Adding a perlin noise texture -> [texture.hpp](Chapter4/texture.hpp)
> 3. Rendering perlin textured spheres -> [perlin_spheres.cpp](Chapter5/perlin_spheres.cpp)
> 4. Getting a smoother perlin texture with linear interpolation
> 5. Adding Hermitian smoothing to limit Mach Band Effect
> 6. Adding a noise scale to the perlin texture
> 7. Replacing points with random unit vectors
> 8. Simulating turbulence with perlin noise
> 9. Rendering marble like spheres with perlin noise -> [perlin_spheres.cpp](Chapter5/marble_spheres.cpp)

> * Chapter 6:
> 1. Modelating image as a texture and using texture coordinates to refer pixels in the image -> [texture.hpp](Chapter4/texture.hpp)
> 2. Configuring [stb_image](https://github.com/nothings/stb) utility in our header file -> [stb_image.h](Chapter6/rtw_stb_image.hpp)
> 3. Rendering a scene with an image texture -> [earth_sphere.cpp](Chapter6/earth_sphere.cpp)

> * Chapter 7:
> 1. Modelating emissive material to obtain light sources -> [material.hpp](Chapter2/material.hpp)
> 2. Adding a black background to the scene so that only light sources contribute to the final image
> 3. Introducing axis aligned xy rectangles -> [aarect.hpp](Chapter7/aarect.hpp)
> 4. Rendering a scene with a rectangle light source -> [aarect_light.cpp](Chapter7/aarect_light.cpp)
> 5. Adding the other 2 axis aligned rectangles, xz, yz -> [aarect.hpp](Chapter7/aarect.hpp)
> 6. Modelating a Cornell Box with our axis aligned rectangles -> [cornell_box.cpp](Chapter7/cornell_box.cpp)