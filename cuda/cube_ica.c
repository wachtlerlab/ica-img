
#include "cube.h"
#include "cube_blas.h"
#include "cube_private.h"
#include "cube_ica.h"
//#include "cube_ica_kernels.h"
#include "calc_z.h"

void
cube_ica_calc_Z (cube_t        *ctx,
		 cube_matrix_t *S,
		 cube_matrix_t *mu,
		 cube_matrix_t *beta,
		 cube_matrix_t *sigma,
		 cube_matrix_t *Z)
{
  
  gpu_calc_Z (S->dev_ptr,
	      S->m,
	      S->n,
	      mu->dev_ptr,
	      beta->dev_ptr,
	      sigma->dev_ptr,
	      Z->dev_ptr);
}

cube_matrix_t *
cube_ica_update_A (cube_t        *ctx,
		   cube_matrix_t *A,
		   cube_matrix_t *S,
		   cube_matrix_t *mu,
		   cube_matrix_t *beta,
		   cube_matrix_t *sigma,
		   const double  *epsilon)
{
  return NULL;
}
