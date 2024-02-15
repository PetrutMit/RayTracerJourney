.PHONY=clean

CC=mpic++

final_scene_build: final_scene.cpp
	$(CC) -o final_scene final_scene.cpp

final_scene_run: final_scene_build
	mpirun -np 4 ./final_scene 

clean:
	rm -f final_scene