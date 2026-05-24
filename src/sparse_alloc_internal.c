#include "sparse_alloc_internal.h"
#include <stdlib.h>

sparse_err_t sparse_malloc_array(size_t count, size_t elem_size, void **out) {
    size_t bytes = 0;
    if (!out)
        return SPARSE_ERR_NULL;

    *out = NULL;
    if (count == 0 || elem_size == 0)
        return SPARSE_OK;
    if (sparse_count_bytes_overflow(count, elem_size, &bytes))
        return SPARSE_ERR_ALLOC;

    *out = malloc(bytes);
    if (!*out)
        return SPARSE_ERR_ALLOC;
    return SPARSE_OK;
}

sparse_err_t sparse_calloc_array(size_t count, size_t elem_size, void **out) {
    size_t bytes = 0;
    if (!out)
        return SPARSE_ERR_NULL;

    *out = NULL;
    if (count == 0 || elem_size == 0)
        return SPARSE_OK;
    if (sparse_count_bytes_overflow(count, elem_size, &bytes))
        return SPARSE_ERR_ALLOC;

    *out = calloc(1, bytes);
    if (!*out)
        return SPARSE_ERR_ALLOC;
    return SPARSE_OK;
}
