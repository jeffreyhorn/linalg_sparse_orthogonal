#include "sparse_alloc_internal.h"
#include <stdlib.h>

static long sparse_alloc_fail_after = -1;

void sparse_alloc_test_fail_after(long remaining) { sparse_alloc_fail_after = remaining; }

void sparse_alloc_test_reset(void) { sparse_alloc_fail_after = -1; }

static int sparse_alloc_test_should_fail(void) {
    if (sparse_alloc_fail_after < 0)
        return 0;
    if (sparse_alloc_fail_after == 0) {
        sparse_alloc_fail_after = -1;
        return 1;
    }
    sparse_alloc_fail_after--;
    return 0;
}

sparse_err_t sparse_malloc_array(size_t count, size_t elem_size, void **out) {
    size_t bytes = 0;
    if (!out)
        return SPARSE_ERR_NULL;

    *out = NULL;
    if (count == 0 || elem_size == 0)
        return SPARSE_OK;
    if (sparse_count_bytes_overflow(count, elem_size, &bytes))
        return SPARSE_ERR_ALLOC;
    if (sparse_alloc_test_should_fail())
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
    if (sparse_alloc_test_should_fail())
        return SPARSE_ERR_ALLOC;

    *out = calloc(1, bytes);
    if (!*out)
        return SPARSE_ERR_ALLOC;
    return SPARSE_OK;
}

sparse_err_t sparse_malloc_idx_array(idx_t count, size_t elem_size, void **out) {
    size_t size_count = 0;
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (count == 0 || elem_size == 0)
        return count < 0 ? SPARSE_ERR_ALLOC : SPARSE_OK;
    if (sparse_idx_to_size_checked(count, &size_count))
        return SPARSE_ERR_ALLOC;
    return sparse_malloc_array(size_count, elem_size, out);
}

sparse_err_t sparse_calloc_idx_array(idx_t count, size_t elem_size, void **out) {
    size_t size_count = 0;
    if (!out)
        return SPARSE_ERR_NULL;
    *out = NULL;
    if (count == 0 || elem_size == 0)
        return count < 0 ? SPARSE_ERR_ALLOC : SPARSE_OK;
    if (sparse_idx_to_size_checked(count, &size_count))
        return SPARSE_ERR_ALLOC;
    return sparse_calloc_array(size_count, elem_size, out);
}
