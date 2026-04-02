/*
 * example_basic_solve.c — Solve a sparse linear system Ax = b using LU factorization.
 *
 * Demonstrates:
 *   - Creating a sparse matrix programmatically
 *   - LU factorization with partial pivoting
 *   - Forward/backward solve
 *   - Residual computation
 *
 * Build:
 *   cc -O2 -Iinclude -o example_basic_solve examples/example_basic_solve.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== Sparse LU Solve Example ===\n\n");

    /* Create a 5x5 sparse tridiagonal system:
     *   4  -1   0   0   0     x0     3
     *  -1   4  -1   0   0     x1     2
     *   0  -1   4  -1   0  *  x2  =  2
     *   0   0  -1   4  -1     x3     2
     *   0   0   0  -1   4     x4     3
     */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    if (!A) {
        fprintf(stderr, "Failed to create matrix\n");
        return 1;
    }

    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i + 1 < n)
            sparse_insert(A, i, i + 1, -1.0);
    }

    printf("Matrix A (%d x %d, %d nonzeros):\n", (int)n, (int)n, (int)sparse_nnz(A));
    for (idx_t i = 0; i < n; i++) {
        printf("  [");
        for (idx_t j = 0; j < n; j++)
            printf(" %5.1f", sparse_get(A, i, j));
        printf(" ]\n");
    }

    /* Right-hand side */
    double b[] = {3.0, 2.0, 2.0, 2.0, 3.0};
    printf("\nRight-hand side b:\n  [");
    for (idx_t i = 0; i < n; i++)
        printf(" %.1f", b[i]);
    printf(" ]\n");

    /* Make a copy for factorization (LU modifies the matrix in-place) */
    SparseMatrix *LU = sparse_copy(A);
    if (!LU) {
        fprintf(stderr, "Failed to copy matrix\n");
        sparse_free(A);
        return 1;
    }

    /* Factor with partial pivoting */
    sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-14);
    if (err != SPARSE_OK) {
        fprintf(stderr, "LU factorization failed (err=%d)\n", (int)err);
        sparse_free(A);
        sparse_free(LU);
        return 1;
    }
    printf("\nLU factorization: OK (nnz after = %d)\n", (int)sparse_nnz(LU));

    /* Solve */
    double x[5];
    err = sparse_lu_solve(LU, b, x);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Solve failed (err=%d)\n", (int)err);
        sparse_free(A);
        sparse_free(LU);
        return 1;
    }

    printf("\nSolution x:\n  [");
    for (idx_t i = 0; i < n; i++)
        printf(" %8.5f", x[i]);
    printf(" ]\n");

    /* Compute residual: r = b - A*x */
    double Ax[5] = {0};
    sparse_matvec(A, x, Ax);
    double resid = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = b[i] - Ax[i];
        resid += ri * ri;
    }
    resid = sqrt(resid);
    printf("\nResidual ||b - Ax|| = %.2e\n", resid);

    /* Expected solution: x = [1, 1, 1, 1, 1] */
    printf("Expected:  x = [ 1.00000  1.00000  1.00000  1.00000  1.00000 ]\n");

    sparse_free(A);
    sparse_free(LU);
    return 0;
}
