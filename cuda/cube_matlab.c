#include "cube.h"
#include "cube_blas.h"
#include "cube_private.h"
#include "cube_matlab.h"


#include <mat.h>

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


