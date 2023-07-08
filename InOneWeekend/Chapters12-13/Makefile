.PHONY=clean

CC=g++
CFLAGS=-Wall -Wextra -pedantic -std=c++17 -g

depth_of_field: depth_of_field.cpp
	$(CC) $(CFLAGS) -o depth_of_field depth_of_field.cpp

final_scene: final_scene.cpp
	$(CC) $(CFLAGS) -o final_scene final_scene.cpp

clean:
	rm -f depth_of_field final_scene