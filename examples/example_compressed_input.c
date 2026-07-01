/*
 * example_compressed_input.c - Build from caller-owned CSR data and solve.
 *
 * Demonstrates:
 *   - Creating a public SparseMatrix shell from caller-owned CSR arrays
 *   - Keeping the input arrays caller-owned after construction
 *   - Using the normal one-shot LU workflow after compressed construction
 *
 * Build:
 *   cc -O2 -Iinclude -o example_compressed_input \
 *      examples/example_compressed_input.c -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_csr.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    printf("=== Compressed Input Example ===\n\n");

    idx_t row_ptr[] = {0, 2, 5, 8, 11, 13};
    idx_t col_idx[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    double values[] = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0};

    SparseCsr csr = {
        .rows = 5,
        .cols = 5,
        .nnz = 13,
        .row_ptr = row_ptr,
        .col_idx = col_idx,
        .values = values,
    };

    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_from_csr(&csr, &A);
    if (err != SPARSE_OK) {
        fprintf(stderr, "CSR construction failed: %s\n", sparse_strerror(err));
        return 1;
    }

    /* The CSR arrays remain caller-owned; A is an independent matrix shell. */
    values[0] = 99.0;
    printf("A(0,0) after caller-owned CSR mutation: %.1f\n", sparse_get(A, 0, 0));
    values[0] = 4.0;

    double b[] = {3.0, 2.0, 2.0, 2.0, 3.0};
    double x[5];

    SparseMatrix *LU = sparse_copy(A);
    if (!LU) {
        fprintf(stderr, "Failed to copy matrix\n");
        sparse_free(A);
        return 1;
    }

    err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-14);
    if (err != SPARSE_OK) {
        fprintf(stderr, "LU factorization failed: %s\n", sparse_strerror(err));
        sparse_free(LU);
        sparse_free(A);
        return 1;
    }

    err = sparse_lu_solve(LU, b, x);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Solve failed: %s\n", sparse_strerror(err));
        sparse_free(LU);
        sparse_free(A);
        return 1;
    }

    printf("Solution x:\n  [");
    for (idx_t i = 0; i < 5; i++)
        printf(" %8.5f", x[i]);
    printf(" ]\n");

    double Ax[5] = {0};
    sparse_matvec(A, x, Ax);
    double resid = 0.0;
    for (idx_t i = 0; i < 5; i++) {
        double ri = b[i] - Ax[i];
        resid += ri * ri;
    }
    printf("Residual ||b - Ax|| = %.2e\n", sqrt(resid));
    printf("Expected:  x = [ 1.00000  1.00000  1.00000  1.00000  1.00000 ]\n");

    sparse_free(LU);
    sparse_free(A);
    return 0;
}
