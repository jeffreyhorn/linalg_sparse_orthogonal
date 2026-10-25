#ifndef SPARSE_INSTALLED_WIDTH_CHECK_H
#define SPARSE_INSTALLED_WIDTH_CHECK_H

#include <limits.h>
#include <sparse/sparse_types.h>
#include <stdio.h>

static int sparse_installed_width_check(size_t *archive_bits_out) {
    const size_t consumer_bits = sizeof(idx_t) * CHAR_BIT;
    const size_t archive_bits = sparse_idx_bits();
    if (consumer_bits != SPARSE_BUILT_IDX_BITS || archive_bits != SPARSE_BUILT_IDX_BITS) {
        fprintf(stderr, "Sparse index width mismatch: consumer=%zu header=%d archive=%zu\n",
                consumer_bits, SPARSE_BUILT_IDX_BITS, archive_bits);
        return 0;
    }
    if (archive_bits_out)
        *archive_bits_out = archive_bits;
    return 1;
}

#endif
