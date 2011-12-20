
#include "cube.h"
#include "cube_blas.h"
#include "cube_matlab.h"
#include "cube_ica.h"
#include "cube_ica_kernels.h"

#include <matrix.h>
#include <mat.h>

#include <string.h>

typedef struct NAPair {

  const char    *name;
  mxArray      **pa;

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
  mxArray *dA, *A, *S, *Z, *mu, *beta, *sigma, *X;
  mxArray *mxa;
  NAPair  *iter;
  NAPair a_map[] = {"A", &A, "S", &S, "mu", &mu, "beta", &beta, "sigma", &sigma, NULL,};
  double a, b, s;
  int maxidx;
  double epsilon;
  double *eps;
  int *iamax;
  int res;
  char *str;

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
	    *(iter->pa) = a;
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

  epsilon = 0.0400;
  res = cube_matlab_ica_update_A (ctx, A, S, mu, beta, sigma, epsilon);
  printf ("res = %d\n", res);

  //str = mxArrayToString (A);
  //printf ("\nA=\n\n%s\n\n", str);
  //mxFree (str);

  
    //cube_matrix_sync (ctx, A, CUBE_SYNC_HOST);
    //cube_matrix_dump (A, 10, 10);
  //cube_matrix_dump (A, 10, 10);

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
