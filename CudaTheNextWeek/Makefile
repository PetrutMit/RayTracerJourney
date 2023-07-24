.PHONY: clean

CC=nvcc
DEBUG=
FLAGS=lineinfo

movingSpheres: movingSpheres.cu
	$(CC) $^ -o $@

checkerSpheres: checkerSpheres.cu
	$(CC) $^ -o $@

clean:
	rm -f movingSpheres