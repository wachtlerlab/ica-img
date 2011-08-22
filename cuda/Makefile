
CC=clang
NVCC=nvcc
GPUARCH=sm_13
CFLAGS=-I/usr/local/cuda/include
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lm -lcudart

%.o: %.c
	$(CC) $(CFLAGS) -g -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CFLAGS) -g -G -arch $(GPUARCH) -c -o $@ $< 

%.ptx: %.cu
	$(NVCC) $(CFLAGS) -m 32 -arch $(GPUARCH) -ptx $< 

absmax: absmax.o absmax_kernel.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

kernels: absmax_kernel.ptx calc_z.ptx

.PHONY: clean

clean:
	rm *.o absmax
