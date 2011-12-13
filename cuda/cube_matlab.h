#ifndef CUBE_MATLAB_H
#define CUBE_MATLAB_H

#include <matrix.h>

#include "cube_matrix.h"

typedef struct _cube_matfile_t cube_matfile_t;

cube_matfile_t * cube_matfile_open  (cube_t *ctx, const char *file);
void             cube_matfile_close (cube_t *ctx, cube_matfile_t *fd);
const char **    cube_matfile_get_dir (cube_t *ctx, cube_matfile_t *fd, int *n);
mxArray *        cube_matfile_get_var (cube_t *ctx, cube_matfile_t *fd, const char *name);
cube_matrix_t *  cube_matrix_from_array (cube_t *ctx, mxArray *array);


#endif
