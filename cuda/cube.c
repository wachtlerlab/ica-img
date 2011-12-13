
#include "cube.h"
#include "cube_error.h"
#include "cube_private.h"

#include <string.h>

static cube_t ctx_no_mem = {CUBE_ERROR_NO_MEMORY, CUBLAS_STATUS_NOT_INITIALIZED, 0};

cube_t *
cube_context_new ()
{
  cube_t *ctx;

  ctx = malloc (sizeof(cube_t));

  if (ctx == NULL)
    return &ctx_no_mem;

  memset (ctx, 0, sizeof (cube_t));

  ctx->e_blas = cublasCreate(&(ctx->h_blas));

  if (ctx->e_blas != CUBLAS_STATUS_SUCCESS)
    ctx->status = CUBE_ERROR_BLAS;

  return ctx;
}

void
cube_context_destroy (cube_t **ctx)
{
  cube_t *ctxp;

  if (ctx == NULL || *ctx == NULL)
    return;

  ctxp = *ctx;
  *ctx = NULL;

  if (ctxp == &ctx_no_mem)
    return;

  if (ctxp->e_blas != CUBLAS_STATUS_NOT_INITIALIZED)
    cublasDestroy(ctxp->h_blas);

  free (ctxp);
}

int
cube_context_check (cube_t *ctx)
{
  if (ctx == NULL)
    return 0;

  if (ctx->status != CUBE_STATUS_OK)
    return 0;
  
  return 1;
}

int
cube_blas_check (cube_t *ctx, cublasStatus_t blas_status)
{
  if (blas_status == CUBLAS_STATUS_SUCCESS)
    return 1;

  ctx->status = CUBE_ERROR_CUDA;
  ctx->e_blas = blas_status;

  return 0;
}

int
cube_cuda_check (cube_t *ctx, cudaError_t cuda_error)
{
  if (cuda_error == cudaSuccess)
      return 1;
  
  ctx->status = CUBE_ERROR_CUDA;
  ctx->e_cuda = cuda_error;

  return 0;
}

void *
cube_malloc_device (cube_t *ctx, size_t size)
{
  cudaError_t res; 
  void *dev_ptr;

  if (! cube_context_check (ctx))
    return NULL;

  res = cudaMalloc (&dev_ptr, size);

  if (! cube_cuda_check (ctx, res))
    dev_ptr = NULL;

  return dev_ptr;
}


void
cube_free_device (cube_t *ctx, void *dev_ptr)
{
  if (! cube_context_check (ctx))
    return;

  cudaFree(dev_ptr);
}


void *
cube_memcpy (cube_t *ctx,
	     void   *dest,
	     void   *src,
	     size_t  n,
	     cube_memcpy_kind_t kind)
{
  cudaError_t res;

  if (! cube_context_check (ctx))
    return NULL;

  res = cudaMemcpy (dest, src, n, (enum cudaMemcpyKind) kind);

  if (! cube_cuda_check (ctx, res))
    dest = NULL;

  return dest;
}

