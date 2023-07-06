.PHONY=clean

CC=g++
CFLAGS=-Wall -Wextra -pedantic -std=c++17 -g

vfov_camera: vfov_camera.cpp
	$(CC) $(CFLAGS) -o vfov_camera vfov_camera.cpp

viewpoints: viewpoints.cpp
	$(CC) $(CFLAGS) -o viewpoints viewpoints.cpp

clean:
	rm -f vfov_camera viewpoints