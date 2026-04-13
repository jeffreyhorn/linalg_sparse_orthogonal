/*
 * example_minnorm.c — Minimum-norm least-squares for underdetermined systems.
 *
 * Demonstrates:
 *   - Building an underdetermined system (m < n)
 *   - Computing the minimum 2-norm solution via sparse_qr_solve_minnorm
 *   - Comparing with a non-minimum-norm solution
 *   - Iterative refinement
 *   - Rank diagnostics
 *
 * Build:
 *   cc -O2 -Iinclude -o example_minnorm examples/example_minnorm.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double vec_norm2(const double *x, int n) {
    double s = 0;
    for (int i = 0; i < n; i++)
        s += x[i] * x[i];
    return sqrt(s);
}

int main(void) {
    printf("=== Minimum-Norm Least Squares Example ===\n\n");

    /* Build a 3x6 underdetermined system:
     * A = [2 0 0 1 0 0]   b = [3]
     *     [0 3 0 0 1 0]       [4]
     *     [0 0 1 0 0 2]       [5]
     */
    int m = 3, n = 6;
    SparseMatrix *A = sparse_create(m, n);
    if (!A) {
        fprintf(stderr, "Failed to create matrix\n");
        return 1;
    }
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 4, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 2, 5, 2.0);

    double b[3] = {3.0, 4.0, 5.0};
    double x[6];

    printf("System: %dx%d (underdetermined)\n\n", m, n);

    /* Compute minimum-norm solution */
    sparse_err_t err = sparse_qr_solve_minnorm(A, b, x, NULL);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Minimum-norm solve failed: %s\n", sparse_strerror(err));
        sparse_free(A);
        return 1;
    }

    printf("Minimum-norm solution:\n  x = [");
    for (int i = 0; i < n; i++)
        printf("%s%.4f", i > 0 ? ", " : "", x[i]);
    printf("]\n  ||x||_2 = %.4f\n\n", vec_norm2(x, n));

    /* Verify A*x = b */
    double Ax[3] = {0};
    sparse_matvec(A, x, Ax);
    printf("Verification: A*x = [%.4f, %.4f, %.4f]\n", Ax[0], Ax[1], Ax[2]);
    printf("Expected b   = [%.4f, %.4f, %.4f]\n\n", b[0], b[1], b[2]);

    /* Compare with a non-minimum-norm solution */
    /* x_alt = [1.5, 4/3, 5, 0, 0, 0] is also a solution but has larger norm */
    double x_alt[6] = {1.5, 4.0 / 3.0, 5.0, 0, 0, 0};
    printf("Alternative solution (not minimum-norm):\n  x_alt = [");
    for (int i = 0; i < n; i++)
        printf("%s%.4f", i > 0 ? ", " : "", x_alt[i]);
    printf("]\n  ||x_alt||_2 = %.4f\n\n", vec_norm2(x_alt, n));

    printf("||x_min|| = %.4f < ||x_alt|| = %.4f\n\n", vec_norm2(x, n), vec_norm2(x_alt, n));

    /* Iterative refinement */
    double residual;
    err = sparse_qr_refine_minnorm(A, b, x, 3, &residual, NULL);
    if (err == SPARSE_OK) {
        printf("After refinement: residual = %.2e\n", residual);
    }

    sparse_free(A);
    printf("\nDone.\n");
    return 0;
}
