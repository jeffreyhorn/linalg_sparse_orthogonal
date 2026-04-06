/**
 * CMake integration example — demonstrates find_package(Sparse) usage.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_PREFIX_PATH=<sparse-install-prefix>
 *   cmake --build .
 *   ./example
 */
#include <sparse/sparse_types.h>
#include <sparse/sparse_matrix.h>
#include <sparse/sparse_lu.h>
#include <sparse/sparse_lu_csr.h>
#include <stdio.h>

int main(void) {
    printf("Sparse library version %s (int %d)\n",
           SPARSE_VERSION_STRING, SPARSE_VERSION);

    /* Create a small 3x3 SPD system and solve with LU */
    SparseMatrix *A = sparse_create(3, 3);
    if (!A) {
        fprintf(stderr, "sparse_create failed\n");
        return 1;
    }

    /* Diagonal-dominant matrix */
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);

    double b[3] = {5.0, 6.0, 5.0};
    double x[3] = {0};

    sparse_err_t err = lu_csr_factor_solve(A, b, x, 1e-12);
    if (err != SPARSE_OK) {
        fprintf(stderr, "solve failed: %s\n", sparse_strerror(err));
        sparse_free(A);
        return 1;
    }

    printf("Solution: [%.6f, %.6f, %.6f]\n", x[0], x[1], x[2]);
    sparse_free(A);
    printf("OK\n");
    return 0;
}
