
#include "cube.h"
#include "cube_blas.h"
#include "cube_private.h"
#include "cube_ica.h"
#include "cube_ica_kernels.h"
#include "calc_z.h"

void
cube_ica_calc_Z (cube_t        *ctx,
		 cube_matrix_t *S,
		 cube_matrix_t *mu,
		 cube_matrix_t *beta,
		 cube_matrix_t *sigma,
		 cube_matrix_t *Z)
{
  if (! cube_context_check (ctx))
    return;

  gpu_calc_Z (S->dev_ptr,
	      S->m,
	      S->n,
	      mu->dev_ptr,
	      beta->dev_ptr,
	      sigma->dev_ptr,
	      Z->dev_ptr);
}

void
cube_ica_update_A (cube_t        *ctx,
		   cube_matrix_t *A,
		   cube_matrix_t *S,
		   cube_matrix_t *mu,
		   cube_matrix_t *beta,
		   cube_matrix_t *sigma,
		   const double  *npats,
		   const double  *epsilon)
{
  cube_matrix_t *Z, *dA, *X;
  double a, b, s;
  int maxidx, *iamax;
  int m, n;

  if (! cube_context_check (ctx))
    return;

  if (!A || !S || !mu || !beta || !sigma || !npats || !epsilon)
    return;

  m = cube_matrix_get_m (S);
  n = cube_matrix_get_n (S);
  Z = cube_matrix_new_on_device (ctx, m, n);

  X = cube_matrix_new_on_device (ctx, m, n);

  m = cube_matrix_get_m (A);
  n = cube_matrix_get_n (A);
  dA = cube_matrix_new_on_device (ctx, m, n);

  /** 1) **/
  cube_ica_calc_Z (ctx, S, mu, beta, sigma, Z);

  
  /** 2) **/
  a = -1.0;
  b = 0.0;

  /*  X = -1*A*Z  */
  cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_N, &a, A, Z, &b, X);

  /* dA = X * S' - npats * A" (with A" = copy of A) */
  a = 1.0;
  b = -100.0;
  cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_T, &a, X, S, &b, dA);
 
  //s = 1.0/100.0 ; /* FIXME: parameter */
  cube_matrix_scale (ctx, dA, npats);

  /** 3) **/
  cube_matrix_iamax (ctx, dA, &maxidx);

  //eps = cube_host_register (ctx, &epsilon, sizeof (epsilon));
  iamax = cube_host_register (ctx, &maxidx, sizeof (maxidx));

  gpu_update_A_with_delta_A (ctx, A, dA, epsilon, iamax);

  

  /* free Z, free dA */
}
