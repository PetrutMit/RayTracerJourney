## From a Ray Traced Image to a Ray Traced Scene

Until this point the project ray traced some stunning images. Rendering times
dropped significantly and the quality of the images increased. Now it is time to add
some interactivity.
This can be attained with the help of `OpenGL` and a `first person camera`. The central
idea is to compute the scene using the CUDA kernel and display the result using OpenGL.
This can be done by rendering the image to a texture and then mapping this texture to a
quad. The quad is then rendered to the screen.
<hr>

For portability reasons, the project will use `CMake` as a build system. This will allow
to compile the project in different platforms and IDEs.

## Structure
* 
<hr>
