#### Mitrache Mihnea

# CUDA Implementation for Ray Tracing The Next Week

> Steps taken to make our CUDA implementation:
> 1. Strating from CUDA In One Weekend implementation and adding the new functionalities
> 2. Actualizing files to support time stamps in rays
> 3. Rendering previous final scene with motion -> [movingSpheres.cu](movingSpheres.cu)
> 4. Because of time considerations and casts, spheres which do not move are created as
moving spheres with same position in time
> 5. Adding a AABB cuda header file -> [aabb.cuh](aabb.cuh)
> 6. Adding a BVH cuda header file -> [bvh.cuh](bvh.cuh)
> 7. Adding a texture class -> [texture.cuh](texture.cuh)
> 8. Modelating solid color texture
> 9. Modelating checker texture
> 10. Rendering spheres with checked ground -> [checkerSpheres.cu](checkerSpheres.cu)