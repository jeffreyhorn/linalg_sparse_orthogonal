/*
 * example_least_squares.c — Solve an overdetermined system via QR factorization.
 *
 * Demonstrates:
 *   - Creating a tall (m > n) sparse matrix
 *   - Column-pivoted QR factorization
 *   - Least-squares solve: minimize ||Ax - b||
 *   - Residual and rank reporting
 *
 * Build:
 *   cc -O2 -Iinclude -o example_least_squares examples/example_least_squares.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== Least Squares via QR Example ===\n\n");

    /* Overdetermined system: 6 equations, 3 unknowns.
     * A = [1 0 0]   b = [1.1]
     *     [0 1 0]       [2.0]
     *     [0 0 1]       [3.1]
     *     [1 1 0]       [2.9]
     *     [0 1 1]       [5.2]
     *     [1 0 1]       [4.0]
     *
     * The system is inconsistent (noisy measurements).
     * QR gives the least-squares solution. */
    idx_t m = 6, n = 3;
    SparseMatrix *A = sparse_create(m, n);
    if (!A) {
        fprintf(stderr, "Failed to create matrix\n");
        return 1;
    }

    /* Row 0: [1 0 0] */
    sparse_insert(A, 0, 0, 1.0);
    /* Row 1: [0 1 0] */
    sparse_insert(A, 1, 1, 1.0);
    /* Row 2: [0 0 1] */
    sparse_insert(A, 2, 2, 1.0);
    /* Row 3: [1 1 0] */
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    /* Row 4: [0 1 1] */
    sparse_insert(A, 4, 1, 1.0);
    sparse_insert(A, 4, 2, 1.0);
    /* Row 5: [1 0 1] */
    sparse_insert(A, 5, 0, 1.0);
    sparse_insert(A, 5, 2, 1.0);

    double b[] = {1.1, 2.0, 3.1, 2.9, 5.2, 4.0};

    printf("System: %d equations, %d unknowns, %d nonzeros\n", (int)m, (int)n, (int)sparse_nnz(A));
    printf("A:\n");
    for (idx_t i = 0; i < m; i++) {
        printf("  [");
        for (idx_t j = 0; j < n; j++)
            printf(" %4.1f", sparse_get(A, i, j));
        printf(" ]   b[%d] = %4.1f\n", (int)i, b[i]);
    }

    /* QR factorization */
    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    if (err != SPARSE_OK) {
        fprintf(stderr, "QR factorization failed (err=%d)\n", (int)err);
        sparse_free(A);
        return 1;
    }

    /* Numerical rank */
    idx_t rank = sparse_qr_rank(&qr, 0.0);
    printf("\nQR factorization: OK (rank = %d)\n", (int)rank);

    /* Least-squares solve */
    double x[3];
    double residual_norm = 0.0;
    err = sparse_qr_solve(&qr, b, x, &residual_norm);
    if (err != SPARSE_OK) {
        fprintf(stderr, "QR solve failed (err=%d)\n", (int)err);
        sparse_qr_free(&qr);
        sparse_free(A);
        return 1;
    }

    printf("\nLeast-squares solution x:\n  [");
    for (idx_t j = 0; j < n; j++)
        printf(" %8.5f", x[j]);
    printf(" ]\n");
    printf("Residual norm ||Ax - b||: %.4f\n", residual_norm);

    /* Verify: compute Ax and show per-equation residuals */
    printf("\nPer-equation residuals:\n");
    for (idx_t i = 0; i < m; i++) {
        double ax = 0.0;
        for (idx_t j = 0; j < n; j++)
            ax += sparse_get(A, i, j) * x[j];
        printf("  eq %d: Ax=%.4f, b=%.4f, residual=%.4f\n", (int)i, ax, b[i], b[i] - ax);
    }

    sparse_qr_free(&qr);
    sparse_free(A);
    return 0;
}
