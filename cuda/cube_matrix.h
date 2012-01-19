#ifndef CUBE_MATRIX_H
#define CUBE_MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _cube_matrix_t cube_matrix_t;

struct _cube_matrix_t {
  
  int     m;
  int     n;

  double *data;
  void   *dev_ptr;
  
  char    order;
};

enum _cube_sync_dir_t {

  CUBE_SYNC_HOST = 0,
  CUBE_SYNC_DEVICE = 1
};

typedef enum _cube_sync_dir_t cube_sync_dir_t;

cube_matrix_t * cube_matrix_new_from_data (cube_t       *ctx,
					   int           m,
					   int           n,
					   double       *data,
					   char          order);

cube_matrix_t * cube_matrix_new_on_device (cube_t       *ctx,
					   int           m,
					   int           n);

void            cube_matrix_copy         (cube_t          *ctx,
					  cube_matrix_t   *x,
					  cube_matrix_t   *y,
					  cube_sync_dir_t  where);


void            cube_matrix_destroy (cube_t        *ctx,
				     cube_matrix_t *matrix);

void            cube_matrix_sync    (cube_t          *ctx,
				     cube_matrix_t   *matrix,
				     cube_sync_dir_t  direction);

int             cube_matrix_get_m   (cube_matrix_t *matrix);
int             cube_matrix_get_n   (cube_matrix_t *matrix);

void            cube_matrix_dump    (cube_matrix_t *matrix, int m_max, int n_max);

void            cube_matrix_iamax   (cube_t              *ctx,
				     const cube_matrix_t *A,
				     int                 *result);
void
cube_matrix_gemm (cube_t *ctx,
		  cube_blas_op_t transa, cube_blas_op_t transb,
		  const double *alpha,
		  const cube_matrix_t *A,
		  const cube_matrix_t *B,
		  const double *beta,
		  cube_matrix_t *C);

void
cube_matrix_scale (cube_t          *ctx,
		   cube_matrix_t   *x,
		   const double    *alpha);

#ifdef __cplusplus
}
#endif
			     
#endif
