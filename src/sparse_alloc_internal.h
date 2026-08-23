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

static inline int sparse_idx_to_size_checked(idx_t value, size_t *out) {
    if (!out)
        return 1;
    if (value < 0)
        return 1;
    if ((uintmax_t)value > SIZE_MAX)
        return 1;
    *out = (size_t)value;
    return 0;
}

static inline int sparse_idx_count_bytes_overflow(idx_t count, size_t elem_size, size_t *bytes) {
    size_t size_count = 0;
    if (sparse_idx_to_size_checked(count, &size_count))
        return 1;
    return sparse_count_bytes_overflow(size_count, elem_size, bytes);
}

static inline int sparse_size_to_idx_checked(size_t value, idx_t *out) {
    if (!out)
        return 1;
    if ((uintmax_t)value > (uintmax_t)IDX_MAX)
        return 1;
    *out = (idx_t)value;
    return 0;
}

sparse_err_t sparse_malloc_array(size_t count, size_t elem_size, void **out);
sparse_err_t sparse_calloc_array(size_t count, size_t elem_size, void **out);
sparse_err_t sparse_malloc_idx_array(idx_t count, size_t elem_size, void **out);
sparse_err_t sparse_calloc_idx_array(idx_t count, size_t elem_size, void **out);

void sparse_alloc_test_fail_after(long remaining);
void sparse_alloc_test_reset(void);

#endif /* SPARSE_ALLOC_INTERNAL_H */
