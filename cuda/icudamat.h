

#ifndef ICUDAMAT_H
#define ICUDAMAT_H

typedef struct _CUDAHandle CUDAHandle;


CUDAHandle * icudamat_create (void);
int          icudamat_idamax (CUDAHandle *handle, const mxArray *main);
int          icudamat_dgemm  (CUDAHandle *handle,
			      mxArray    *A,	
			      int         transa,
			      mxArray    *B,
			      int         transb,
			      mxArray    *C,
			      double      alpha,
			      double      beta);
#endif
