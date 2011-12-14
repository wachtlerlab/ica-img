
#include "cube.h"
#include "cube_blas.h"
#include "cube_matlab.h"
#include "cube_ica.h"
#include "cube_ica_kernels.h"

#include <matrix.h>
#include <mat.h>

#include <string.h>

typedef struct NAPair {

  const char     *name;
  cube_matrix_t **pa;

} NAPair;



int
main(int argc, char **argv)
{
  cube_t  *ctx;
  cube_matfile_t *fd;
  MATFile *fmat;
  const char **dir;
  char *filename;
  int i, ndir;
  cube_matrix_t *dA, *A, *S, *Z, *mu, *beta, *sigma, *X;
  mxArray *mxa;
  NAPair  *iter;
  NAPair a_map[] = {"A", &A, "S", &S, "mu", &mu, "beta", &beta, "sigma", &sigma, NULL,};
  double a, b, s;
  int maxidx;
  double epsilon;
  double *eps;
  int *iamax;

  dA = A = S = mu = beta = sigma = Z = NULL;

  ctx = cube_context_new ();

  filename = argv[1];

  fd = cube_matfile_open (ctx, filename);
  dir = cube_matfile_get_dir (ctx, fd, &ndir);

  for (i = 0; i < ndir; i++)
    {
      const char *name = dir[i];
      mxArray    *a;

      a = cube_matfile_get_var (ctx, fd, name);
 
      for (iter = a_map; iter->name; iter++)
	if (strcmp (iter->name, name) == 0)
	  {
	    cube_matrix_t *cm;

	    cm = cube_matrix_from_array (ctx, a);
	    cube_matrix_sync (ctx, cm, CUBE_SYNC_DEVICE);

	    *(iter->pa) = cm;
	    a = NULL;
	    printf ("%s -> %s\n", iter->name, name);
	    break;
	  }
      
      if (a != NULL)
	{
	  printf ("Unused: %s \n", name);
	  mxDestroyArray (a);
	}
    }

  cube_matrix_dump (A, 10, 10);

  mxa = mxCreateDoubleMatrix (147, 100, mxREAL);
  Z = cube_matrix_from_array (ctx, mxa);
  cube_matrix_sync (ctx, Z, CUBE_SYNC_DEVICE);

  X = cube_matrix_new_on_device (ctx, 147, 100);

  mxa = mxCreateDoubleMatrix (147, 147, mxREAL);
  dA = cube_matrix_from_array (ctx, mxa);
  cube_matrix_sync (ctx, dA, CUBE_SYNC_DEVICE);
  //dA = cube_matrix_new_on_device (ctx, 147, 147);
  cube_matrix_copy (ctx, A, dA, CUBE_SYNC_DEVICE);

  cube_ica_calc_Z (ctx, S, mu, beta, sigma, Z);

  a = -1.0;
  b = 0.0;

  /*  X = -1*A*Z  */
  cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_N, &a, A, Z, &b, X);

  /* dA = X * S' - 40000 * A" (with A" = copy of A) */
  a = 1.0;
  b = -100.0;
  cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_T, &a, X, S, &b, dA);
 
  s = 1.0/100.0;
  cube_matrix_scale (ctx, dA, &s);

  /*cube_matrix_dump (A);
  cube_matrix_dump (dA);
  cube_matrix_dump (MA);
  */

  //cube_matrix_gemm (ctx, CUBE_BLAS_OP_N, CUBE_BLAS_OP_N, &a, A, dA, &b, MA);

  cube_matrix_sync (ctx, Z, CUBE_SYNC_HOST);
  cube_matrix_dump (Z, 10, 10);
  //str = mxArrayToString (Z->data)

  //cube_matrix_sync (ctx, Z, CUBE_SYNC_HOST);
  //cube_matrix_dump (Z, 10, 10);

  cube_matrix_iamax (ctx, dA, &maxidx);

  cube_matrix_sync (ctx, dA, CUBE_SYNC_HOST);
  cube_matrix_dump (dA, 10, 10);

  printf ("amax: %d\n",  maxidx);

  epsilon = 0.0400;

  eps = cube_host_register (ctx, &epsilon, sizeof (epsilon));
  iamax = cube_host_register (ctx, &maxidx, sizeof (maxidx));

  gpu_update_A_with_delta_A (ctx, A, dA, eps, iamax);

  cube_matrix_sync (ctx, A, CUBE_SYNC_HOST);
  cube_matrix_dump (A, 10, 10);


  //for (iter = a_map; iter->name; iter++)
  // {
  //    mxArray *a;
  //    if ((a = *(iter->pa)) != NULL)
  //	mxDestroyArray (a);
  //   }
  /*
  cube_matrix_destroy (ctx, cdA);
  cube_matrix_destroy (ctx, cA);
  cube_matrix_destroy (ctx, cMA);
  */

  cube_matfile_close (ctx, fd);
  cube_context_destroy (&ctx);

  return 0;
}
