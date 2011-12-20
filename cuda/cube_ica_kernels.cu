
#include "cube.h"
#include "cube_blas.h"
#include "cube_matrix.h"

#include "cube_ica_kernels.h"
#include "cube_private.h"

#include <cuda.h>
#include <stdio.h>


__device__ double dsign (double v)
{
  if (v > 0)
    return 1;
  else if (v < 0)
    return -1;
  else
    return 0;
}

__global__ void
kernel_calc_z (const double *S_g,
	       int           m,
	       int           n,
	       const double *mu_g,
	       const double *beta_g,
	       const double *sigma_g,
	       double       *Z)
{
  extern __shared__ double smem[];
  double mu, beta, sigma;
  double *mu_s, *beta_s, *sigma_s, *S_s, *Z_s;
  double s, q, c, z;
  int    global_x, global_y, lid, gid;

    /* calculate global and local ids */
  global_x = (blockDim.x * blockIdx.x) + threadIdx.x;
  global_y = (blockDim.y * blockIdx.y) + threadIdx.y;

  if (global_x > n)
    return;

  gid = (n * global_y) + global_x;
  lid = (threadIdx.y * blockDim.x) + threadIdx.x;

  mu_s = &smem[0];
  beta_s = &smem[blockDim.x];
  sigma_s = &smem[2*blockDim.x];
  S_s = &smem[3 * blockDim.x];
  Z_s = &smem[(3 + blockDim.y) *blockDim.x];

  mu_s[threadIdx.y] = mu_g[global_y];
  beta_s[threadIdx.y] = beta_g[global_y];
  sigma_s[threadIdx.y] = sigma_g[global_y];

  S_s[lid] = S_g[gid];

  __syncthreads();
  
  if (global_y > m)
    return;

  mu = mu_s[threadIdx.y];
  beta = beta_s[threadIdx.y];
  sigma = sigma_s[threadIdx.y];

  s = S_s[lid];

  /* do the computation */
  s -= mu;
  q = (2.0/(1.0+beta));
  c = pow ((tgamma(3.0/q)/tgamma(1.0/q)), (q/2.0));
  z = -1 * (q*c/pow (sigma,q)) * pow (abs (s), q-1.0) * dsign (s);

  Z_s[lid] = z;

  __syncthreads();

  Z[gid] = Z_s[lid]; 
}


int
cube_gpu_calc_Z (cube_t        *ctx,
		 cube_matrix_t *S,
		 cube_matrix_t *Z,
		 cube_matrix_t *mu,
		 cube_matrix_t *beta,
		 cube_matrix_t *sigma)
{
  cudaError_t res;
  double *devS, *devZ, *devmu, *devbeta, *devsigma;
  dim3 grid, block;
  int m, n;
  size_t smem;

  if (! cube_context_check (ctx))
    return cube_context_check (ctx);

  m = cube_matrix_get_m (Z);
  n = cube_matrix_get_n (Z);

  block.x = 16;
  block.y = 16;

  grid.x = ceil (n / (double) block.x);
  grid.y = ceil (m / (double) block.y);

  smem = block.y * sizeof (double) * (3 + 2*block.x);

  devS = (double *) S->dev_ptr;
  devZ = (double *) Z->dev_ptr;
  devmu = (double *) mu->dev_ptr;
  devbeta = (double *) beta->dev_ptr;
  devsigma = (double *) sigma->dev_ptr;

  kernel_calc_z<<<grid, block, smem>>>(devS, m, n, devmu, devbeta, devsigma, devZ);

  res = cudaPeekAtLastError ();
  return cube_cuda_check (ctx, res);
}

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

  dA_data[lid] = dA[gid];
  A_data[lid] = A[gid];

  __syncthreads();

  /* do the computation */
  max = abs(dA[*iamax - 1]); /* global read, but LDU hopefully (FIXME, not sure) */
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

  block.z = grid.z = 1;

  smem = 2 * block.x * block.y * sizeof (double);

  devA = (double *) A->dev_ptr;
  devdA = (double *) dA->dev_ptr;

  printf ("%u, %u, %zu\n", grid.x, grid.y, smem);

  update_AdA_kernel<<<grid, block, smem>>>(devA, devdA, m, n, epsilon, iamax);
  res = cudaPeekAtLastError ();

  cube_cuda_check (ctx, res);
}
