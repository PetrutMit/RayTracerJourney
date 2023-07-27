.PHONY: clean

CC=nvcc
DEBUG=-g -G
FLAGS=-lineinfo

movingSpheres: movingSpheres.cu
	$(CC) $^ -o $@

checkerSpheres: checkerSpheres.cu
	$(CC) $(FLAGS) $^ -o $@

perlinSpheres: perlinSpheres.cu
	$(CC) $^ -o $@

test_random: test_random.cu
	$(CC) $^ -o $@

light: light.cu
	$(CC) $^ -o $@

cornellBox: cornellBox.cu
	$(CC) $^ -o $@

cornellBoxWithBoxes: cornellBoxWithBoxes.cu
	$(CC) $^ -o $@

cornellBoxWithTransformedBoxes: cornellBoxWithTransformedBoxes.cu
	$(CC) $^ -o $@

cornellBoxWithSmoke: cornellBoxWithSmoke.cu
	$(CC) $^ -o $@

clean:
	rm -f movingSpheres checkerSpheres perlinSpheres test_random light cornellBox cornellBoxWithBoxes cornellBoxWithTransformedBoxes cornellBoxWithSmoke
