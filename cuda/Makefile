MATLAB=/usr/local/MATLAB/R2011a
CC=clang
NVCC=nvcc
GPUARCH=sm_13
ARCH=$(shell getconf LONG_BIT)
CUDA=/usr/local/cuda
CFLAGS=-I$(CUDA)/include -I$(MATLAB)/extern/include -m$(ARCH) -g
LIBS=-lcuda -lm -lcudart -lcublas -lmex -lmx
SO_EXT=so

ifeq ($(ARCH), 32)
	LDFLAGS=-L$(CUDA)/lib -L$(MATLAB)/bin/glnx86
else
	LDFLAGS=-L$(CUDA)/lib$(ARCH) -L$(MATLAB)/bin/glnxa64
endif


%.o: %.c
	$(CC) $(CFLAGS) -fPIC -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CFLAGS) -G -arch $(GPUARCH) -c -o $@ $< 

%.ptx: %.cu
	$(NVCC) $(CFLAGS) -arch $(GPUARCH) -ptx $< 

all : icudamat.$(SO_EXT)

absmax: absmax.o absmax_kernel.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

icudamat.$(SO_EXT): icudamat.o
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS) $(LIBS)

kernels: absmax_kernel.ptx calc_z.ptx

transpose: transpose.o


.PHONY: clean

clean:
	rm -f *.o absmax icudamat.$(SO_EXT)
