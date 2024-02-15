.PHONY: clean

CC=nvcc

final_scene: final_scene.cu
	$(CC) $(FLAGS) -o $@ $^

clean:
	rm -f final_scene



