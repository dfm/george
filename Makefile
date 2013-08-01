CC = g++
CFLAGS = -Igeorge -I/usr/local/include/eigen3

.cpp.o:
	$(CC) $(CFLAGS) -o $*.o -c $*.cpp

test: george/test.cpp george/george.h
	$(CC) $(CFLAGS) george/test.cpp -o test

clean:
	rm -rf test george/*.o
