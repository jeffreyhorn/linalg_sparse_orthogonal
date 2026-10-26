#include "sparse_types.h"

#include <limits.h>
#include <stdio.h>

_Static_assert(sizeof(idx_t) * CHAR_BIT == SPARSE_BUILT_IDX_BITS,
               "idx_t must match the generated library width");

int main(void) {
    size_t typedef_bits = sizeof(idx_t) * CHAR_BIT;
    size_t header_bits = SPARSE_BUILT_IDX_BITS;
    size_t archive_bits = sparse_idx_bits();

    if (archive_bits != header_bits) {
        fprintf(stderr, "index width mismatch: typedef=%zu header=%zu archive=%zu\n", typedef_bits,
                header_bits, archive_bits);
        return 1;
    }

    printf("index width: typedef=%zu header=%zu archive=%zu\n", typedef_bits, header_bits,
           archive_bits);
    return 0;
}
