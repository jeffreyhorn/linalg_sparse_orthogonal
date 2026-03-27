/*
 * Quick smoke test: create matrix, factor, solve, check residual.
 * This is NOT the unit test framework — just a build/link verification.
 */
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    /* 3x3 test matrix:
     *   1  0  3
     *   0  5  0
     *   7  0  9
     */
    SparseMatrix *A = sparse_create(3, 3);
    if (!A) {
        fprintf(stderr, "create failed\n");
        return 1;
    }

    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    /* Copy before factoring (for residual check) */
    SparseMatrix *A_orig = sparse_copy(A);
    if (!A_orig) {
        fprintf(stderr, "copy failed\n");
        sparse_free(A);
        return 1;
    }

    printf("Original matrix:\n");
    sparse_print_dense(A, stdout);
    sparse_print_info(A, stdout);

    /* Factor with complete pivoting */
    sparse_err_t err = sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-10);
    if (err != SPARSE_OK) {
        fprintf(stderr, "LU factor failed: %s\n", sparse_strerror(err));
        sparse_free(A);
        sparse_free(A_orig);
        return 1;
    }

    printf("\nAfter LU (L+U in logical view):\n");
    sparse_print_dense(A, stdout);

    /* Solve A*x = b */
    double b[3] = {1.0, 2.0, 3.0};
    double x[3] = {0};

    err = sparse_lu_solve(A, b, x);
    if (err != SPARSE_OK) {
        fprintf(stderr, "solve failed: %s\n", sparse_strerror(err));
        sparse_free(A);
        sparse_free(A_orig);
        return 1;
    }

    printf("\nSolution x = [%.6f, %.6f, %.6f]\n", x[0], x[1], x[2]);

    /* Residual check: r = A_orig * x - b */
    double r[3];
    sparse_matvec(A_orig, x, r);
    double max_residual = 0.0;
    for (int i = 0; i < 3; i++) {
        r[i] -= b[i];
        double ar = fabs(r[i]);
        if (ar > max_residual)
            max_residual = ar;
    }
    printf("Max residual: %.3e\n", max_residual);

    /* Iterative refinement */
    err = sparse_lu_refine(A_orig, A, b, x, 3, 1e-15);
    if (err != SPARSE_OK) {
        fprintf(stderr, "refine failed: %s\n", sparse_strerror(err));
    }

    sparse_matvec(A_orig, x, r);
    max_residual = 0.0;
    for (int i = 0; i < 3; i++) {
        r[i] -= b[i];
        double ar = fabs(r[i]);
        if (ar > max_residual)
            max_residual = ar;
    }
    printf("Max residual after refinement: %.3e\n", max_residual);

    /* Test partial pivoting too */
    SparseMatrix *B = sparse_copy(A_orig);
    err = sparse_lu_factor(B, SPARSE_PIVOT_PARTIAL, 1e-10);
    if (err != SPARSE_OK) {
        fprintf(stderr, "partial pivot LU failed: %s\n", sparse_strerror(err));
    } else {
        double x2[3] = {0};
        sparse_lu_solve(B, b, x2);
        printf("\nPartial pivot solution x = [%.6f, %.6f, %.6f]\n", x2[0], x2[1], x2[2]);
    }
    sparse_free(B);

    /* Matrix Market round-trip */
    err = sparse_save_mm(A_orig, "/tmp/smoke_test.mtx");
    if (err == SPARSE_OK) {
        SparseMatrix *loaded = NULL;
        err = sparse_load_mm(&loaded, "/tmp/smoke_test.mtx");
        if (err == SPARSE_OK) {
            printf("\nMM round-trip: loaded nnz = %d (expected %d)\n", (int)sparse_nnz(loaded),
                   (int)sparse_nnz(A_orig));
            sparse_free(loaded);
        }
    }

    int pass = (max_residual < 1e-10);
    printf("\n%s\n", pass ? "SMOKE TEST PASSED" : "SMOKE TEST FAILED");

    sparse_free(A);
    sparse_free(A_orig);
    return pass ? 0 : 1;
}
