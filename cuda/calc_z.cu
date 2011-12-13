
#include "calc_z.h"
#include <cuda.h>

__device__ double sign (double v)
{
  if (v > 0)
    return 1;
  else if (v < 0)
    return -1;
  else
    return 0;
}

__global__ void calc_z (const double *full_s,
			const double *mp_mu,
			const double *mp_beta,
			const double *mp_sigma,
			double       *zout)
{
  double mu, beta, sigma;
  int gid, tid;

  mu = mp_mu[blockIdx.y];
  beta = mp_beta[blockIdx.y];
  sigma = mp_sigma[blockIdx.y];

  tid = threadIdx.x;
  gid = (blockIdx.y * blockDim.x) + tid;
  
  if (tid < blockDim.x)
    {
      double s, q, c, z;

      s = full_s[gid];
      s -= mu;
      q = (2.0/(1.0+beta));
      c = pow ((tgamma(3.0/q)/tgamma(1.0/q)), (q/2.0));
      z = -1 * (q*c/pow (sigma,q)) * pow (abs (s), q-1.0) * sign (s);
      zout[gid] = z;
    }
}


void
gpu_calc_Z (const double *S,
	    int           m,
	    int           n,
	    const double *mu,
	    const double *beta,
	    const double *sigma,
	    double *Z)
{
  dim3 grid;
  dim3 block;

  grid.y = m;
  block.x = n;

  calc_z<<<grid, block>>>(S, mu, beta, sigma, Z);
}
