.PHONY: clean

CC=nvcc
FLAGS=-lineinfo

movingSpheres: movingSpheres.cu
	$(CC) $^ -o $@

checkerSpheres: checkerSpheres.cu
	$(CC) $(FLAGS) $^ -o $@

perlinSpheres: perlinSpheres.cu
	$(CC) $^ -o $@

test_random: test_random.cu
	$(CC) $^ -o $@

clean:
	rm -f movingSpheres checkerSpheres perlinSpheres test_random