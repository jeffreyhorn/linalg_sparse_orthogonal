#ifndef SPARSE_ALLOC_INTERNAL_H
#define SPARSE_ALLOC_INTERNAL_H

/*
 * Private header: shared internal allocation/overflow helpers for
 * size arithmetic and array allocation. Not part of the public API.
 */

#include "sparse_types.h"
#include <stddef.h>
#include <stdint.h>

static inline int sparse_size_mul_overflow(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a)
        return 1;
    *out = a * b;
    return 0;
}

static inline int sparse_size_add_overflow(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b)
        return 1;
    *out = a + b;
    return 0;
}

static inline int sparse_count_bytes_overflow(size_t count, size_t elem_size, size_t *bytes) {
    return sparse_size_mul_overflow(count, elem_size, bytes);
}

static inline int sparse_idx_count_bytes_overflow(idx_t count, size_t elem_size, size_t *bytes) {
    if (count < 0)
        return 1;
    return sparse_count_bytes_overflow((size_t)count, elem_size, bytes);
}

static inline int sparse_size_to_idx_checked(size_t value, idx_t *out) {
    if (value > (size_t)IDX_MAX)
        return 1;
    *out = (idx_t)value;
    return 0;
}

sparse_err_t sparse_malloc_array(size_t count, size_t elem_size, void **out);
sparse_err_t sparse_calloc_array(size_t count, size_t elem_size, void **out);

#endif /* SPARSE_ALLOC_INTERNAL_H */
