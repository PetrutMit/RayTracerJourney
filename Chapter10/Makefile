.PHONY=clean

CC=g++
CFLAGS=-Wall -Wextra -pedantic -std=c++17 -g

dielectric_spheres: dielectric_spheres.cpp
	$(CC) $(CFLAGS) -o dielectric_spheres dielectric_spheres.cpp

hollow_glass_spheres: hollow_glass_spheres.cpp
	$(CC) $(CFLAGS) -o hollow_glass_spheres hollow_glass_spheres.cpp

clean:
	rm -f dielectric_spheres hollow_glass_spheres
