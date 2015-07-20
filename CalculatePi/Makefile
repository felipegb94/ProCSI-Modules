CC=clang
OPTS=-std=c11 -fopenmp -o
EXECUTABLES=pi-omp pi-serial 


pi-omp: pi-omp.c
	$(CC) $(OPTS) pi-omp pi-omp.c

pi-serial: pi-serial.c
	$(CC) $(OPTS) pi-serial pi-serial.c

all:
	make clean
	make $(EXECUTABLES)

clean:
	rm -f $(EXECUTABLES)

