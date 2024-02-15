.PHONY=clean

CC=mpic++
FLAGS=-fopenmp

final_scene_build: final_scene.cpp
	$(CC) $(FLAGS) final_scene.cpp -o final_scene

final_scene_run: final_scene_build
	mpirun -np 4 ./final_scene

clean:
	rm -f final_scene