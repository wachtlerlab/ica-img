MATLAB=/usr/local/MATLAB/R2011a
CC=clang
NVCC=nvcc
GPUARCH=sm_13
ARCH=$(shell getconf LONG_BIT)
CUDA=/usr/local/cuda
CFLAGS=-I$(CUDA)/include -I$(MATLAB)/extern/include -m$(ARCH) -g
LIBS=-lcuda -lm -lcudart -lcublas
SO_EXT=so

ifeq ($(ARCH), 32)
	MATLAB_BINDIR=$(MATLAB)/bin/glnx86
	CUDA_LIBDIR=$(CUDA)/lib
else
	MATLAB_BINDIR=$(MATLAB)/bin/glnxa64
	CUDA_LIBDIR=$(CUDA)/lib64
endif

LDFLAGS=-L$(CUDA_LIBDIR) -L$(MATLAB_BINDIR) -L. -Wl,-rpath,$(MATLAB_BINDIR) -Wl,-rpath,$(CUDA_LIBDIR) -Wl,-rpath,.

%.o: %.c
	$(CC) $(CFLAGS) -fPIC -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CFLAGS) -G -arch $(GPUARCH) -c -o $@ $< 

%.ptx: %.cu
	$(NVCC) $(CFLAGS) -arch $(GPUARCH) -ptx $< 

all : libcube.$(SO_EXT)

absmax: absmax.o absmax_kernel.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

libcube.$(SO_EXT): cube.o cube_blas.o cube_matrix.o cube_ica.o calc_z.o
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS) $(LIBS)

libcube_m.$(SO_EXT): libcube.$(SO_EXT) cube_matlab.o
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS) $(LIBS) -lcube -lmat -lmx

kernels: absmax_kernel.ptx calc_z.ptx

transpose: transpose.o

updateA: updateA.o libcube_m.$(SO_EXT)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) -lcube -lmat -lmx

.PHONY: clean

clean:
	rm -f *.o absmax icudamat.$(SO_EXT) libcube.$(SO_EXT)
