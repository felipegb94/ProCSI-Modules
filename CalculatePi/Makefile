CC=gcc
OPTS=-std=c11 -lm -fopenmp -o
EXECUTABLES=pi-serial-rectangles pi-serial-montecarlo pi-openmp-rectangles pi-openmp-rectangles2 pi-openmp-montecarlo 


pi-serial-rectangles: pi-serial-rectangles.c
	$(CC) $(OPTS) pi-serial-rectangles pi-serial-rectangles.c

pi-serial-montecarlo: pi-serial-montecarlo.c
	$(CC) $(OPTS) pi-serial-montecarlo pi-serial-montecarlo.c

pi-openmp-rectangles: pi-openmp-rectangles.c
	$(CC) $(OPTS) pi-openmp-rectangles  pi-openmp-rectangles.c

pi-openmp-rectangles2: pi-openmp-rectangles2.c
	$(CC) $(OPTS) pi-openmp-rectangles2  pi-openmp-rectangles2.c

pi-openmp-montecarlo: pi-openmp-montecarlo.c
	$(CC) $(OPTS) pi-openmp-montecarlo pi-openmp-montecarlo.c


all:
	make clean
	make $(EXECUTABLES)

clean:
	rm -f $(EXECUTABLES)

