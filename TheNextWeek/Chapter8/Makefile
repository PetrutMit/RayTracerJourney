.PHONY:clean

CC=g++
CFLAGS=-Wall -Wextra
DEBUG=-g

aarect_light: aarect_light.cpp
	$(CC) $(CFLAGS) -o aarect_light aarect_light.cpp

cornell_box_with_boxes: cornell_box_with_boxes.cpp
	$(CC) $(CFLAGS) -o cornell_box_with_boxes cornell_box_with_boxes.cpp

cornell_box_with_tr_boxes: cornell_box_with_tr_boxes.cpp
	$(CC) $(CFLAGS) $(DEBUG) -o cornell_box_with_tr_boxes cornell_box_with_tr_boxes.cpp

clean:
	rm -f cornell_box_with_boxes cornell_box_with_tr_boxes