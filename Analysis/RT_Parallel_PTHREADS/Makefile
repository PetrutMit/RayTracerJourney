.PHONY=clean

CC=g++
CFLAGS=-std=c++17 -pthread -lpthread

final_scene: final_scene.cpp thread_pool_o
	$(CC) $(CFLAGS) -o final_scene final_scene.cpp thpool.o

thread_pool_o: thpool.c
	gcc -c thpool.c

clean:
	rm -f final_scene thpool.o