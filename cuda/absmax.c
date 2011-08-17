
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "absmax_kernel.h"

double *
alloc_matrix(int m, int n)
{
  int     matrix_size;
  size_t  mem_size;
  int     i;
  double *matrix;

  matrix_size = m*n;
  mem_size = sizeof (double) * matrix_size;
  matrix = malloc (mem_size);

  for (i = 0; i < matrix_size; i++)
    matrix[i] = i;

  return matrix;
}

void
dump_matrix (double *matrix, int m, int n)
{
  int i, j;

  for (i = 0; i < m; i++)
    {
      for (j = 0; j < n; j++)
        {
          double val = *(matrix + (i*n+j));
          fprintf (stdout, "%.2f |", val);
        }

      fprintf (stdout, "\n");
    }
}

int
main (int argc, char **argv)
{
  int                    dev_id;
  int                    dev_count;
  struct cudaDeviceProp  dev_props;
  double                *matrix;
  double                 result;
  cudaError_t            res;

  cudaGetDevice (&dev_id);
  cudaGetDeviceProperties (&dev_props, dev_id);

  fprintf (stderr, "Device [%d]: %s\n", dev_id, dev_props.name);

  matrix = alloc_matrix (256, 256);
  //dump_matrix (matrix, 294, 294);

  absmax (matrix, 256*256, &result);
  free (matrix);
  fprintf (stdout, "%f\n", result);

  return 0;
}
