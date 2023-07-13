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

testAllocation: testAllocation.cu
	$(CC) -o $@ $^

multipleSpheresAA: multipleSpheresAA.cu
	$(CC) -o $@ $^

diffuseSpheres: diffuseSpheres.cu
	$(CC) -o $@ $^

metalSpheres: metalSpheres.cu
	$(CC) -o $@ $^

dielectricSpheres: dielectricSpheres.cu
	$(CC) -o $@ $^

viewpoints: viewpoints.cu
	$(CC) -o $@ $^

depthOfField: depthOfField.cu
	$(CC) -o $@ $^

finalScene: finalScene.cu
	$(CC) -o $@ $^

clean:
	rm -f firstCuda gradient simpleSphere surfaceNormals multipleSpheres testAllocation multipleSpheresAA diffuseSpheres metalSpheres dielectricSpheres



