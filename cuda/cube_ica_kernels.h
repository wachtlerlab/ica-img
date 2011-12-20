#ifndef CUDA_ICA_KERNEL_H
#define CUDA_ICA_KERNEL_H

#ifdef __cplusplus
extern "C" {

#endif
int
cube_gpu_calc_Z (cube_t        *ctx,
		 cube_matrix_t *S,
		 cube_matrix_t *Z,
		 cube_matrix_t *mu,
		 cube_matrix_t *beta,
		 cube_matrix_t *sigma);

void
gpu_update_A_with_delta_A (cube_t        *ctx,
			   cube_matrix_t *A,
			   cube_matrix_t *dA,
			   const double  *epsilon,
			   const int     *iamax);

#ifdef __cplusplus
}
#endif

#endif
