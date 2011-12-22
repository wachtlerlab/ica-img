#include "cube.h"
#include "cube_blas.h"
#include "cube_private.h"
#include "cube_matlab.h"
#include "cube_ica.h"

#include "cube_ica_kernels.h"

#include <mat.h>
#include <stdarg.h>

typedef struct MATFile _cube_matfile_t;


cube_matfile_t *
cube_matfile_open (cube_t *ctx, const char *filename)
{
  MATFile *fd;

  if (! cube_context_check (ctx))
    return NULL;

  fd = matOpen (filename, "r");

  if (fd == NULL)
    ctx->status = CUBE_ERROR_FAILED; 

  return (cube_matfile_t *) fd;
}

void
cube_matfile_close (cube_t *ctx, cube_matfile_t *mfd)
{
  MATFile *fd = (MATFile *) mfd;

  if (! cube_context_check (ctx) || fd == NULL)
    return;

  matClose (fd);
}

const char **
cube_matfile_get_dir (cube_t *ctx, cube_matfile_t *mfd, int *n)
{
  MATFile *fd = (MATFile *) mfd;
  const char **dir;
    
  if (! cube_context_check (ctx))
    return NULL;

  dir = (const char **) matGetDir (fd, n);

  if (dir == NULL)
    {
      ctx->status = CUBE_ERROR_FAILED;
      *n = 0;
    }

  return dir;
}

mxArray *
cube_matfile_get_var (cube_t         *ctx,
		      cube_matfile_t *mfd,
		      const char     *name)
{
  MATFile *fd = (MATFile *) mfd;
  mxArray *a;

  if (! cube_context_check (ctx) || fd == NULL || name == NULL)
    return NULL;

  a = matGetVariable (fd, name);

  if (a == NULL)
    ctx->status = CUBE_ERROR_FAILED; 

  return a;
}


int
cube_matfile_get_vars (cube_t         *ctx,
		       cube_matfile_t *mfd,
		       ...)
{
  va_list ap;
  char    *var;
  int      count;

  count = 0;

  va_start (ap, mfd); 

  while ((var = va_arg (ap, char *)) != NULL)
    {
      mxArray **map;

      map = va_arg (ap, mxArray **);

      if (ap == NULL)
	break;

      *map = cube_matfile_get_var (ctx, mfd, var);
      count++;
    }
 
  va_end(ap);
  return count;
}


cube_matrix_t *
cube_matrix_from_array (cube_t *ctx, mxArray *A)
{
  int m, n, ndims;
  void *data;
  cube_matrix_t *matrix;

  if (! cube_context_check (ctx) || A == NULL)
    return NULL;

  if (! mxIsDouble (A))
    return NULL;

  ndims = mxGetNumberOfDimensions (A);
  
  if (ndims != 2)
    return NULL;

  m = (int) mxGetM (A);
  n = (int) mxGetN (A);
  data = mxGetData (A);

  matrix = cube_matrix_new_from_data (ctx, m, n, data, 'F');

  return matrix;
}


int
cube_matlab_ica_update_A (cube_t  *ctx,
			  mxArray *m_A,
			  mxArray *m_S,
			  mxArray *m_mu,
			  mxArray *m_beta,
			  mxArray *m_sigma,
			  double   m_epsilon)
{
  cube_matrix_t *A, *S, *mu, *beta, *sigma;
  double *epsilon, npats;

  if (! cube_context_check (ctx))
    return -1;

  A     = cube_matrix_from_array (ctx, m_A);
  S     = cube_matrix_from_array (ctx, m_S);
  mu    = cube_matrix_from_array (ctx, m_mu);
  beta  = cube_matrix_from_array (ctx, m_beta);
  sigma = cube_matrix_from_array (ctx, m_sigma);

  npats   = 1.0 / (double) cube_matrix_get_n (S);
  epsilon = cube_host_register (ctx, &m_epsilon, sizeof (double));

  cube_matrix_sync (ctx, A, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, S, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, mu, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, beta, CUBE_SYNC_DEVICE);
  cube_matrix_sync (ctx, sigma, CUBE_SYNC_DEVICE);

  cube_ica_update_A (ctx, A, S, mu, beta, sigma, &npats, epsilon);

  cube_matrix_sync (ctx, A, CUBE_SYNC_HOST);

  cube_host_unregister (ctx, &m_epsilon);
  cube_matrix_destroy (ctx, A);
  cube_matrix_destroy (ctx, S);
  cube_matrix_destroy (ctx, mu);
  cube_matrix_destroy (ctx, sigma);

  return cube_context_check (ctx);
}

int
cube_matlab_ica_adapt_prior (cube_t  *ctx,
			     mxArray *Sp,
			     double   mu,
			     double   sigma,
			     double   tol,
			     double   a,
			     double   b,
			     mxArray *beta)
{
  int res;
  if (! cube_context_check (ctx))
    return -1;

  res = gpu_adapt_prior (ctx,
			 mxGetPr(Sp),
			 mxGetM(Sp),
			 mxGetN(Sp),
			 mu,
			 sigma,
			 tol,
			 a,
			 b,
			 mxGetPr(beta));

  return res;
}
