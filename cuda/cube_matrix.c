
#include "cube.h"
#include "cube_blas.h"
#include "cube_private.h"
#include "cube_matrix.h"

#include <stdio.h>
#include <string.h>

//#define IDX2F(i, j, ld) ((((j)-1)∗(ld))+((i)-1))
#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))

static cube_matrix_t *
cube_matrix_new (cube_t *ctx)
{
  cube_matrix_t *ma;

  if (! cube_context_check (ctx))
    return NULL;

  ma = malloc (sizeof (cube_matrix_t));
  memset (ma, 0, sizeof (cube_matrix_t));

  return ma;
}

cube_matrix_t *
cube_matrix_new_from_data (cube_t       *ctx,
			   int           m,
			   int           n,
			   double       *data,
			   char          order)
{
  cube_matrix_t *ma;

  ma = cube_matrix_new (ctx);

  if (ma == NULL)
    return NULL;
  
  ma->data = data;
  ma->m = m;
  ma->n = n;
  ma->order = order;
  ma->dev_ptr = NULL;

  return ma;
}

void
cube_matrix_destroy (cube_t        *ctx,
		     cube_matrix_t *matrix)
{
  if (! cube_context_check (ctx) || matrix == NULL)
    return;

  if (matrix->dev_ptr != NULL)
    cube_free_device (ctx, matrix->dev_ptr);

  free (matrix);
}


void
cube_matrix_sync (cube_t          *ctx,
		  cube_matrix_t   *matrix,
		  cube_sync_dir_t  direction)
{
  size_t size;

  if (! cube_context_check (ctx) || matrix == NULL)
    return;

  size = matrix->m * matrix->n * sizeof (double);

  if (direction == CUBE_SYNC_DEVICE)
    {
      if (matrix->dev_ptr == NULL)
	matrix->dev_ptr = cube_malloc_device (ctx, size);
      
      printf ("M: H->D [%zu]\n", size);
      cube_memcpy (ctx,
		   matrix->dev_ptr,
		   matrix->data,
		   size,
		   CMK_HOST_2_DEVICE);
      

    }
  else
    {
      printf ("M: D->H [%zu]\n", size);
      cube_memcpy (ctx,
		   matrix->data,
		   matrix->dev_ptr,
		   size,
		   CMK_DEVICE_2_HOST);
    }
}

void
cube_matrix_gemm (cube_t *ctx,
		  cube_blas_op_t transa, cube_blas_op_t transb,
		  const double *alpha,
		  const cube_matrix_t *A,
		  const cube_matrix_t *B,
		  const double *beta,
		  cube_matrix_t *C)
{
  void *devA, *devB, *devC;
  int m, n, k;
  int x, y, z;
  int lda, ldb, ldc;

  if (! cube_context_check (ctx))
    return;

  m = C->m;
  n = C->n;

  x = A->m;
  k = A->n;
  
  y = B->m;
  z = B->n;

  devA = A->dev_ptr;
  devB = B->dev_ptr;
  devC = C->dev_ptr;

  cube_blas_d_gemm (ctx,
		    transa, transb,
		    m, n, k,
		    alpha,
		    devA, x,
		    devB, y,
		    beta,
		    devC, m);

}

cube_matrix_t *
cube_matrix_new_on_device (cube_t       *ctx,
			   int           m,
			   int           n)
{
  cube_matrix_t *ma;
  size_t         size;

  ma = cube_matrix_new (ctx);

  if (ma == NULL)
    return NULL;

  size = m * n * sizeof (double);

  ma->m = m;
  ma->n = n;
  ma->dev_ptr = cube_malloc_device (ctx, size);

  if (ma->dev_ptr == NULL)
    {
      free (ma);
      ma = NULL;
    }

  return ma;
}

void
cube_matrix_copy         (cube_t          *ctx,
			  cube_matrix_t   *x,
			  cube_matrix_t   *y,
			  cube_sync_dir_t  where)
{
  int n; 

  if (! cube_context_check (ctx))
    return;

   n = x->m * x->n;

   cube_blas_d_copy (ctx,
		     n,
		     x->dev_ptr, 1,
		     y->dev_ptr, 1);

}

void
cube_matrix_scale (cube_t          *ctx,
		   cube_matrix_t   *x,
		   const double    *alpha)
{
  int n; 
  
  if (! cube_context_check (ctx))
    return;
  
  n = x->m * x->n;

  cube_blas_d_scal (ctx, n,
		    alpha,
		    x->dev_ptr, 1);
}

#define min(X,Y) (((X) < (Y)) ? (X) : (Y))

void
cube_matrix_dump (cube_matrix_t *matrix, int m_max, int n_max)
{
  double *A;
  int m;
  int n;
  int row, col;

  A = matrix->data;

  m = min (matrix->m, m_max);
  n = min (matrix->n, n_max);

  for (row = 0; row < m; row++)
    {
      for (col = 0; col < n; col++)
	{
	  int pos = (col * matrix->m) + row;
	  printf ("%lf ", A[pos]); //A[IDX2F(row, col, n)]);
	}

      printf ("\n");
    }
  printf ("\n");
}
