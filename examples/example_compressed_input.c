/*
 * example_compressed_input.c - Build from caller-owned CSR/CSC data and solve.
 *
 * Demonstrates:
 *   - Creating a public SparseMatrix shell from caller-owned CSR arrays
 *   - Creating a public SparseMatrix shell from caller-owned CSC arrays
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

static int solve_and_report(const char *label, const SparseMatrix *A) {
    double b[] = {3.0, 2.0, 2.0, 2.0, 3.0};
    double x[5];

    SparseMatrix *LU = sparse_copy(A);
    if (!LU) {
        fprintf(stderr, "%s: failed to copy matrix\n", label);
        return 0;
    }

    sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-14);
    if (err != SPARSE_OK) {
        fprintf(stderr, "%s: LU factorization failed: %s\n", label, sparse_strerror(err));
        sparse_free(LU);
        return 0;
    }

    err = sparse_lu_solve(LU, b, x);
    sparse_free(LU);
    if (err != SPARSE_OK) {
        fprintf(stderr, "%s: solve failed: %s\n", label, sparse_strerror(err));
        return 0;
    }

    printf("%s solution x:\n  [", label);
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
    printf("%s residual ||b - Ax|| = %.2e\n", label, sqrt(resid));
    return 1;
}

int main(void) {
    printf("=== Compressed Input Example ===\n\n");

    idx_t row_ptr[] = {0, 2, 5, 8, 11, 13};
    idx_t col_idx[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    double csr_values[] = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0};

    SparseCsr csr = {
        .rows = 5,
        .cols = 5,
        .nnz = 13,
        .row_ptr = row_ptr,
        .col_idx = col_idx,
        .values = csr_values,
    };

    SparseMatrix *A_csr = NULL;
    sparse_err_t err = sparse_from_csr(&csr, &A_csr);
    if (err != SPARSE_OK) {
        fprintf(stderr, "CSR construction failed: %s\n", sparse_strerror(err));
        return 1;
    }

    /* The CSR arrays remain caller-owned; A_csr is an independent matrix shell. */
    csr_values[0] = 99.0;
    printf("CSR A(0,0) after caller-owned CSR mutation: %.1f\n", sparse_get(A_csr, 0, 0));
    csr_values[0] = 4.0;

    idx_t col_ptr[] = {0, 2, 5, 8, 11, 13};
    idx_t row_idx[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    double csc_values[] = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0};

    SparseCsc csc = {
        .rows = 5,
        .cols = 5,
        .nnz = 13,
        .col_ptr = col_ptr,
        .row_idx = row_idx,
        .values = csc_values,
    };

    SparseMatrix *A_csc = sparse_create_from_csc(&csc);
    if (!A_csc) {
        fprintf(stderr, "CSC construction failed\n");
        sparse_free(A_csr);
        return 1;
    }

    /* The CSC arrays remain caller-owned; A_csc is an independent matrix shell. */
    csc_values[0] = 77.0;
    printf("CSC A(0,0) after caller-owned CSC mutation: %.1f\n", sparse_get(A_csc, 0, 0));
    csc_values[0] = 4.0;

    if (!solve_and_report("CSR", A_csr) || !solve_and_report("CSC", A_csc)) {
        sparse_free(A_csc);
        sparse_free(A_csr);
        return 1;
    }

    printf("Expected:  x = [ 1.00000  1.00000  1.00000  1.00000  1.00000 ]\n");

    sparse_free(A_csc);
    sparse_free(A_csr);
    return 0;
}
