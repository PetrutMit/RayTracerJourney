.PHONY=clean

CC=g++
CFLAGS=-Wall -Wextra -Werror -pedantic -std=c++17 -g

gradient: gradient.cpp
	$(CC) $(CFLAGS) -o gradient gradient.cpp

gradient_sphere: gradient_sphere.cpp
	$(CC) $(CFLAGS) -o gradient_sphere gradient_sphere.cpp

clean:
	rm -f gradient gradient_sphere