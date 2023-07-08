.PHONY=clean

CC=g++
CFLAGS=-Wall -Wextra -Werror -pedantic -std=c++17 -g

surface_normals: surface_normals.cpp
	$(CC) $(CFLAGS) -o surface_normals surface_normals.cpp

snormals_sphere_ground: snormals_sphere_ground.cpp
	$(CC) $(CFLAGS) -o snormals_sphere_ground snormals_sphere_ground.cpp

clean:
	rm -f surface_normals snormals_sphere
