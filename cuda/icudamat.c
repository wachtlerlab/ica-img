#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cublas_v2.h>

#include <matrix.h>
#include <mex.h>

#include <stdio.h>

#include "icudamat.h"


struct _CUDAHandle {

  cublasHandle_t hcu;

};

CUDAHandle *
icudamat_create()
{
  CUDAHandle *ch;
  cublasStatus_t status;
  cublasHandle_t hcu;

  ch = malloc (sizeof(CUDAHandle));
  
  status = cublasCreate(&hcu);
  
  if (status != cudaSuccess)
    {
      mexErrMsgTxt ("Failed to set pointer mode\n");
      return NULL;
    }

  ch->hcu = hcu;
  return ch;
}

int
icudamat_idamax(CUDAHandle *handle, const mxArray *main)
{
  cublasStatus_t res;
  cudaError_t    r;
  void *data, *deviceData;
  size_t nelm;
  size_t selm;
  int    ridx;

  data = mxGetData (main);
  nelm = mxGetNumberOfElements (main);
  selm = mxGetElementSize (main);

  mexPrintf("%lu, %lu\n", nelm, selm);

  res = cublasSetPointerMode(handle->hcu, CUBLAS_POINTER_MODE_HOST);

  if (res != cudaSuccess)
    {
      mexErrMsgTxt ("Failed to set pointer mode\n");
      return -1;
    }
  
  r  = cudaMalloc (&deviceData, nelm * selm);
  if (r != cudaSuccess)
    {
      mexErrMsgTxt ("Failed to allocated memory on device\n");
      return -1;
    }

  res = cublasSetVector(nelm, selm, data, 1, deviceData, 1);
  
  if (res != cudaSuccess)
    {
      mexErrMsgTxt ("Failed to copy data to GPU\n");
      return -1;
    }

  res = cublasIdamax(handle->hcu, nelm, deviceData, 1, &ridx);

  if (res != cudaSuccess)
    {
      mexErrMsgTxt ("Failed to do computation on GPU\n");
      return -1;
    }

  return ridx;
}

static int
ma_get_dims2d(mxArray *A, int *m, int *n)
{
  mwSize ndims;

  ndims = mxGetNumberOfDimensions (A);
  
  if (ndims != 2)
    return -1;

  *m = (int) mxGetM (A);
  *n = (int) mxGetN (A);
 
  return 0;
}

cublasOperation_t
op_from_int (int trans)
{
  cublasOperation_t res;
  switch (trans)
    {
    case 0:
      res = CUBLAS_OP_N;
      break;
    case 1:
      res = CUBLAS_OP_T;
      break;
    case 2:
      res = CUBLAS_OP_C;
      break;
    }

  return res;
}

void *
ma_host2gpu (mxArray *ma)
{
  cudaError_t err;
  void *dev, *data;
  size_t selm, sma, nelm;

  nelm = mxGetNumberOfElements (ma);
  selm = mxGetElementSize (ma);

  data = mxGetData (ma);

  sma = selm * nelm;

  err = cudaMalloc (&dev, sma);

  if (err != cudaSuccess)
    mexPrintf("Error allocation device memory\n");

  mexPrintf("%lu, %lu\n", nelm, selm);

  err = cudaMemcpy (dev, data, sma, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
    mexPrintf("Error copying to device\n");

  return dev;
}

int
ma_gpu2host(mxArray *ma, void *dev)
{
  cudaError_t err;
  size_t selm, sma, nelm;
  void *data;

  nelm = mxGetNumberOfElements (ma);
  data = mxGetData (ma);
  selm = mxGetElementSize (ma);

  sma = selm * nelm;
  err = cudaMemcpy (data, dev, sma, cudaMemcpyDeviceToHost);

  return err != cudaSuccess;
}

/* A = m x k, B = k x n, C = m x n */
int
icudamat_dgemm(CUDAHandle *handle,
	       mxArray    *A,	
	       int         transa,
	       mxArray    *B,
	       int         transb,
	       mxArray    *C,
	       double      alpha,
	       double      beta)
{
  int m, n, k, x, y, z;
  int r;
  cublasOperation_t ta, tb;
  void *devA, *devB, *devC;
  cublasStatus_t status;
  
  ta = op_from_int (transa);
  tb = op_from_int (transb);

  r = ma_get_dims2d (C, &m, &n);

  r = ma_get_dims2d (A, &x, &k);

  r = ma_get_dims2d (B, &y, &z);

  mexPrintf("C(m,n) A(x,k) B(y,z) m,n,k = %d %d %d  x,y,z = %d %d %d.\n", m, n, k, x, y, z);

  devA = ma_host2gpu (A);
  devB = ma_host2gpu (B);
  devC = ma_host2gpu (C);

  mexPrintf(".\n");

  status = cublasDgemm (handle->hcu,
			ta, tb,
			m, n, k,
			&alpha, devA, x,
			devB, y,
			&beta, devC, m);


  if (status != cudaSuccess)
    mexPrintf("Error during computation %d\n", status);
  else
    ma_gpu2host(C, devC);

  mexPrintf(".\n");
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
  mexPrintf(".\n");

  return 0;
}
