
#ifndef CUBE_ICA_H
#define CUBE_ICA_H

#include "cube_matrix.h"

void cube_ica_calc_Z (cube_t        *ctx,
		      cube_matrix_t *S,
		      cube_matrix_t *mu,
		      cube_matrix_t *beta,
		      cube_matrix_t *sigma,
		      cube_matrix_t *Z);
#endif
