
#ifndef CUBE_H
#define CUBE_H

#include <sys/types.h>

typedef struct _cube_t cube_t;
typedef enum   _cube_status_t cube_status_t;

typedef enum _cube_memcpy_kind_t cube_memcpy_kind_t;

enum  _cube_memcpy_kind_t {

  CMK_HOST_2_HOST     = 0,
  CMK_HOST_2_DEVICE   = 1,
  CMK_DEVICE_2_HOST   = 2,
  CMK_DEVICE_2_DEVICE = 3,
  CMK_DEFAULT         = 4

};

cube_t * cube_context_new ();
void     cube_context_destroy (cube_t **ctx);

int      cube_context_check (cube_t *ctx);

void *   cube_malloc_device (cube_t *ctx, size_t size);
void     cube_free_device (cube_t *ctx, void *dev_ptr);

void *   cube_memcpy (cube_t *ctx,
		      void   *dest,
		      void   *src,
		      size_t  n,
		      cube_memcpy_kind_t kind);

#endif /* CUBE_H */
