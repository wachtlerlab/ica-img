
#ifndef CUBE_ERROR_H
#define CUBE_ERROR_H

enum _cube_status_t {
  CUBE_STATUS_OK  = 0,

  CUBE_ERROR_CUDA = 1,
  CUBE_ERROR_BLAS = 2,

  CUBE_ERROR_NO_MEMORY = 3,

  CUBE_ERROR_FAILED = 255

};

#endif
