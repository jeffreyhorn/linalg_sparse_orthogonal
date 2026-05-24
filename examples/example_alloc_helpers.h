#ifndef EXAMPLE_ALLOC_HELPERS_H
#define EXAMPLE_ALLOC_HELPERS_H

#include "sparse_types.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static inline sparse_err_t example_check_array_bytes(idx_t count, size_t elem_size) {
    if (count < 0)
        return SPARSE_ERR_BADARG;
    if (elem_size != 0 && (size_t)count > SIZE_MAX / elem_size)
        return SPARSE_ERR_ALLOC;
    return SPARSE_OK;
}

static inline sparse_err_t example_malloc_array(idx_t count, size_t elem_size, void **out) {
    sparse_err_t err;

    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;

    err = example_check_array_bytes(count, elem_size);
    if (err != SPARSE_OK || count == 0)
        return err;

    *out = malloc((size_t)count * elem_size);
    return *out ? SPARSE_OK : SPARSE_ERR_ALLOC;
}

static inline sparse_err_t example_calloc_array(idx_t count, size_t elem_size, void **out) {
    sparse_err_t err;

    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;

    err = example_check_array_bytes(count, elem_size);
    if (err != SPARSE_OK || count == 0)
        return err;

    *out = calloc((size_t)count, elem_size);
    return *out ? SPARSE_OK : SPARSE_ERR_ALLOC;
}

#endif /* EXAMPLE_ALLOC_HELPERS_H */
