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
> 5. Adding a checker texture(chess table like) class -> [checker_texture.hpp](Chapter4/checker_texture.hpp)
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