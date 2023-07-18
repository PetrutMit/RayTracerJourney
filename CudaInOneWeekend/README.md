#### Mitrache Mihnea

# CUDA Implementation for Ray Tracing in One Weekend

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
> 27. Introducing defocus blur -> [depthOfField.cu](Cuda/depthOfField.cu)
> 28. Rendering our final beautiful scene -> [finalScene.cu](Cuda/finalScene.cu)
> 29. Finally defeted a Cpp bug worth mentioning
```Cpp
#define SPHERE_COUNT 22*22 + 3 + 1
cudaMalloc((void**)&d_list, SPHERE_COUNT * sizeof(hittable*))
// Results in cuda error: unspecified launch failure ==> SEGFAULT
// Probably because macro is not expanded correctly
// With debug flags on ??
```
> 30. Instrumenting the code and adding some improvements
> 31. Trying to add a source of light in our scene
> 32. Adding support for emissive materials -> [material.cuh](Cuda/material.cuh)

