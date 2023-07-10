.PHONY: clean

CC=nvcc

firstCuda: firstCuda.cu
	$(CC) -o $@ $^

gradient: gradient.cu
	$(CC) -o $@ $^

simpleSphere: simpleSphere.cu
	$(CC) -o $@ $^

surfaceNormals: surfaceNormals.cu
	$(CC) -o $@ $^

multipleSpheres: multipleSpheres.cu
	$(CC) -o $@ $^

clean:
	rm -f firstCuda gradient simpleSphere surfaceNormals



