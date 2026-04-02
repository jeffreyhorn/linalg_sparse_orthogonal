/*
 * example_matrix_free.c — Matrix-free GMRES with a custom operator.
 *
 * Demonstrates:
 *   - Defining a custom matvec callback (sparse_matvec_fn)
 *   - Solving a system where the matrix is never explicitly formed
 *   - Useful for very large or structured operators
 *
 * The operator is a 1D diffusion operator: A*x[i] = -x[i-1] + 2*x[i] - x[i+1]
 * with Dirichlet boundary conditions. This is equivalent to the tridiagonal
 * matrix [-1, 2, -1] but computed procedurally without storing the matrix.
 *
 * Build:
 *   cc -O2 -Iinclude -o example_matrix_free examples/example_matrix_free.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_iterative.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Custom matvec callback: y = A*x where A is the 1D Laplacian.
 * ctx points to the problem size n (just as an example of using context). */
static sparse_err_t laplacian_matvec(const void *ctx, idx_t n, const double *x, double *y) {
    (void)ctx; /* could use ctx to pass additional parameters */

    for (idx_t i = 0; i < n; i++) {
        double left = (i > 0) ? x[i - 1] : 0.0;
        double right = (i + 1 < n) ? x[i + 1] : 0.0;
        y[i] = -left + 2.0 * x[i] - right;
    }
    return SPARSE_OK;
}

/* Simple diagonal preconditioner: M^{-1} = diag(1/2, 1/2, ..., 1/2) */
static sparse_err_t diag_precond(const void *ctx, idx_t n, const double *r, double *z) {
    (void)ctx;
    for (idx_t i = 0; i < n; i++)
        z[i] = r[i] * 0.5;
    return SPARSE_OK;
}

int main(void) {
    printf("=== Matrix-Free GMRES Example ===\n\n");

    idx_t n = 100;
    printf("Problem: 1D Laplacian, n = %d (matrix never explicitly formed)\n", (int)n);
    printf("Operator: y[i] = -x[i-1] + 2*x[i] - x[i+1]\n\n");

    /* Right-hand side: b such that x_exact = sin(pi*i/(n+1)) */
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    double *x_exact = calloc((size_t)n, sizeof(double));
    if (!b || !x || !x_exact) {
        fprintf(stderr, "Allocation failed\n");
        free(b);
        free(x);
        free(x_exact);
        return 1;
    }

    double pi = 3.14159265358979323846;
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin(pi * (double)(i + 1) / (double)(n + 1));

    /* Compute b = A * x_exact using the callback */
    laplacian_matvec(NULL, n, x_exact, b);

    /* --- Solve without preconditioner --- */
    printf("--- GMRES (no preconditioner) ---\n");
    sparse_gmres_opts_t opts = {.restart = 30, .max_iter = 500, .tol = 1e-10};
    sparse_iter_result_t result;
    memset(x, 0, (size_t)n * sizeof(double));

    sparse_err_t err =
        sparse_solve_gmres_mf(laplacian_matvec, NULL, n, b, x, &opts, NULL, NULL, &result);
    printf("  Status:     %s\n", (err == SPARSE_OK) ? "converged" : "NOT converged");
    printf("  Iterations: %d\n", (int)result.iterations);
    printf("  Residual:   %.2e\n", result.residual_norm);

    double err_norm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = x[i] - x_exact[i];
        err_norm += d * d;
    }
    printf("  ||x - x_exact||: %.2e\n", sqrt(err_norm));

    /* --- Solve with diagonal preconditioner --- */
    printf("\n--- GMRES with diagonal preconditioner ---\n");
    memset(x, 0, (size_t)n * sizeof(double));

    err =
        sparse_solve_gmres_mf(laplacian_matvec, NULL, n, b, x, &opts, diag_precond, NULL, &result);
    printf("  Status:     %s\n", (err == SPARSE_OK) ? "converged" : "NOT converged");
    printf("  Iterations: %d\n", (int)result.iterations);
    printf("  Residual:   %.2e\n", result.residual_norm);

    err_norm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = x[i] - x_exact[i];
        err_norm += d * d;
    }
    printf("  ||x - x_exact||: %.2e\n", sqrt(err_norm));

    /* Show a few solution values */
    printf("\nSolution sample (first 5 and last 5):\n");
    printf("  %-4s  %-12s  %-12s\n", "i", "x_computed", "x_exact");
    for (idx_t i = 0; i < 5; i++)
        printf("  %-4d  %12.8f  %12.8f\n", (int)i, x[i], x_exact[i]);
    printf("  ...\n");
    for (idx_t i = n - 5; i < n; i++)
        printf("  %-4d  %12.8f  %12.8f\n", (int)i, x[i], x_exact[i]);

    free(b);
    free(x);
    free(x_exact);
    return 0;
}
