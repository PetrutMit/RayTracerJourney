.PHONY: clean

CC=nvcc

firstCuda: firstCuda.cu
	$(CC) -o $@ $^

gradient: gradient.cu
	$(CC) -o $@ $^

clean:
	rm -f firstCuda gradient


