
#include <cuda.h>
#include <stdio.h>
#include "absmax_kernel.h"

// kernel functions

__global__ void absmax_kernel (const double *in, double *out)
{
  int tid, gid, off;
  extern __shared__ double data[];

  tid = threadIdx.x;
  gid = (blockIdx.x * blockDim.x * 2) + tid*2;
  off = 1;

  // read data from global memory
  data[tid*2] = in[gid];
  data[tid*2+1] = in[gid+1];

  // b-tree result calculation
  for (off = 1; off < blockDim.x*2; off = off << 1)
    {
      __syncthreads ();

      if (tid < (blockDim.x/off))
        {
          int off_x = (tid * off * 2);
	  int off_y = off_x + off;

          data[off_x] = max (abs (data[off_x]), abs (data[off_y]));
        }
    }

  // write memory back to device memory
  __syncthreads ();
  if (tid == 0)
    out[blockIdx.x] = data[0];

}

#if 0
// utility funtions

int
nearest_power_of_two (int v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

int
calc_cfg (int s, int *blocksPerGrid, int *threadsPerBlock, size_t *sharedMemSize)
{
  int p2 = nearest_power_of_two (s);
  int block_size;
  int n_blocks;

  block_size = p2 < 1024 ? p2 : 1024;
  n_blocks = s / block_size;

  *threadsPerBlock = block_size / 2;
  *blocksPerGrid = n_blocks;
  *sharedMemSize = block_size * sizeof (double);

  printf ("cfg: [%d: %d * %d] <<<%d, %d, %ld>>>\n", p2, block_size, n_blocks,
	  *blocksPerGrid, *threadsPerBlock, *sharedMemSize);

  return *blocksPerGrid;
}

// main entry point for the abs max function
void
absmax (const double *matrix, int s, double *result)
{
  int blocksPerGrid;
  int threadsPerBlock;
  size_t sharedMemSize;
  int rem;
  size_t p2;
  double *data;
  double *out;
  double *test;

  p2 = nearest_power_of_two (s);
  cudaMalloc (&data, p2 * sizeof (double));
  cudaMemcpy (data, matrix, s * sizeof (double), cudaMemcpyHostToDevice);
  cudaMemset (data + s, 0, (p2 - s) * sizeof (double));

  rem = calc_cfg (p2, &blocksPerGrid, &threadsPerBlock, &sharedMemSize);
  cudaMalloc (&out, blocksPerGrid * sizeof (double));

  absmax_kernel<<<rem, threadsPerBlock, sharedMemSize>>>(data, out);
  cudaThreadSynchronize ();

  cudaMallocHost (&test, blocksPerGrid * sizeof (double));
  cudaMemcpy (test, out, blocksPerGrid * sizeof (double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < blocksPerGrid; i++)
    printf ("%f | ", test[i]);

  rem = calc_cfg (rem, &blocksPerGrid, &threadsPerBlock, &sharedMemSize);
  data = out;

  

  do {
    cudaThreadSynchronize ();
    absmax_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(data, out);
    rem = calc_cfg (rem, &blocksPerGrid, &threadsPerBlock, &sharedMemSize);
    printf ("REM %d\n", rem);
  } while (rem > 1);

  cudaMemcpy (result, out, sizeof (double), cudaMemcpyDeviceToHost);
}

#endif
