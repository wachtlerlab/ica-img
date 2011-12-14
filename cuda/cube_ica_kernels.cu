
#include "cube.h"
#include "cube_blas.h"
#include "cube_matrix.h"

#include "cube_ica_kernels.h"
#include "cube_private.h"

#include <cuda.h>
#include <stdio.h>

__global__ void
update_AdA_kernel (double       *A,    
		   const double *dA,
		   int           m,
		   int           n,
		   const double *epsilon,
		   const int    *iamax)
{
  double max;
  const double eps = *epsilon;
  extern __shared__ double smem[];
  double *dA_data;
  double *A_data;
  int     global_x, global_y, lid, gid;
  double  temp;
  
  /* calculate global and local ids */
  global_x = (blockDim.x * blockIdx.x) + threadIdx.x;
  global_y = (blockDim.y * blockIdx.y) + threadIdx.y;

  /* see if we are inside the boundaries */
  if (global_x > n || global_y > m)
    return;

  gid = (n * global_y)  + global_x;
  lid = (threadIdx.y * blockDim.x) + threadIdx.x;

  /* set up shared memory addresses  */
  A_data = &smem[0];
  dA_data = &smem[blockDim.x * blockDim.y];

  /* read in the data from global memory */
  max = abs(dA[*iamax - 1]); /* global read, but LDU hopefully (FIXME, not sure) */
  //eps  = *epsilon;

  temp = dA[gid];
  dA_data[lid] = temp;

  A_data[lid] = A[gid];

  __syncthreads();

  /* do the computation */
  A_data[lid] += (eps / max) * dA_data[lid];


  /* write result back */
  __syncthreads();
  A[gid] = A_data[lid];

}

void
gpu_update_A_with_delta_A (cube_t        *ctx,
			   cube_matrix_t *A,
			   cube_matrix_t *dA,
			   const double  *epsilon,
			   const int     *iamax)
{
  cudaError_t res;
  double *devA, *devdA;
  dim3 grid, block;
  size_t smem;
  int  m, n;

  if (! cube_context_check (ctx))
    return;

  m = A->m;
  n = A->n;

  block.x = 16;
  block.y = 16;

  grid.x = ceil (n / (double) block.x);
  grid.y = ceil (m / (double) block.y);

  smem = 2 * block.x * block.y * sizeof (double);

  devA = (double *) A->dev_ptr;
  devdA = (double *) dA->dev_ptr;

  printf ("%u, %u, %zu\n", grid.x, grid.y, smem);

  update_AdA_kernel<<<grid, block, smem>>>(devA, devdA, m, n, epsilon, iamax);
  res = cudaPeekAtLastError ();

  cube_cuda_check (ctx, res);
}
