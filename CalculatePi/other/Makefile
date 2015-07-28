# Enable CUDA
CUDA_ENABLED=0


# Compiler
NVCC=nvcc

OPTS= -lm -o

# OpenMP OPTS
OMP_OPTS=-fopenmp

# CUDA OPTS
CUDA_OPTS=-arch sm_20



EXECUTABLES=pi-cuda-openmp-rectangles
#pi-serial-rectangles pi-serial-montecarlo pi-openmp-rectangles pi-openmp-rectangles2 pi-openmp-montecarlo 


pi-cuda-openmp-rectangles: pi-cuda-openmp-rectangles.c
	$(CC) -DCUDA_ENABLED=$(CUDA_ENABLED) $(OPTS) pi-cuda-openmp-rectangles pi-cuda-openmp-rectangles.c $(OMP_OPTS) $(CUDA_OPTS) 

# pi-serial-rectangles: pi-serial-rectangles.c
# 	$(CC) -DCUDA_ENABLED=$(CUDA_ENABLED) pi-serial-rectangles pi-serial-rectangles.c $(OMP_OPTS) $(CUDA_OPTS) 

# pi-serial-montecarlo: pi-serial-montecarlo.c
# 	$(CC) -DCUDA_ENABLED=$(CUDA_ENABLED) $(OPTS) pi-serial-montecarlo pi-serial-montecarlo.c #$(OMP_OPTS) $(CUDA_OPTS) 

# pi-openmp-rectangles: pi-openmp-rectangles.c
# 	$(CC) -DCUDA_ENABLED=$(CUDA_ENABLED) pi-openmp-rectangles pi-openmp-rectangles.c $(OMP_OPTS) $(CUDA_OPTS) 

# pi-openmp-rectangles2: pi-openmp-rectangles2.c
# 	$(CC) -DCUDA_ENABLED=$(CUDA_ENABLED) pi-openmp-rectangles2 pi-openmp-rectangles2.c $(OMP_OPTS) $(CUDA_OPTS) 

# pi-openmp-montecarlo: pi-openmp-montecarlo.c
# 	$(CC) -DCUDA_ENABLED=$(CUDA_ENABLED) pi-openmp-montecarlo pi-openmp-montecarlo.c $(OMP_OPTS) $(CUDA_OPTS) 

all:
	make clean
	make $(EXECUTABLES)

clean:
	rm -f $(EXECUTABLES)

