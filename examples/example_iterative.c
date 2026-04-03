/*
 * example_iterative.c — Preconditioned GMRES for a large sparse system.
 *
 * Demonstrates:
 *   - Creating a large sparse system
 *   - ILU(0) preconditioner
 *   - GMRES with left preconditioning
 *   - Convergence monitoring
 *   - Comparison: unpreconditioned vs preconditioned
 *
 * Build:
 *   cc -O2 -Iinclude -o example_iterative examples/example_iterative.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build a diagonally-dominant sparse matrix (2D Laplacian-like) */
static SparseMatrix *build_laplacian(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;

    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i + 1 < n)
            sparse_insert(A, i, i + 1, -1.0);
        /* Add a longer-range coupling for a more interesting spectrum */
        idx_t stride = (idx_t)sqrt((double)n);
        if (stride > 1) {
            if (i >= stride)
                sparse_insert(A, i, i - stride, -0.5);
            if (i + stride < n)
                sparse_insert(A, i, i + stride, -0.5);
        }
    }
    return A;
}

int main(void) {
    printf("=== Preconditioned GMRES Example ===\n\n");

    idx_t n = 200;
    SparseMatrix *A = build_laplacian(n);
    if (!A) {
        fprintf(stderr, "Failed to create matrix\n");
        return 1;
    }
    printf("Matrix: %d x %d, %d nonzeros\n", (int)n, (int)n, (int)sparse_nnz(A));

    /* Right-hand side: b = A * [1, 1, ..., 1]^T (so x_exact = [1, 1, ..., 1]) */
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    double *ones = calloc((size_t)n, sizeof(double));
    if (!b || !x || !ones) {
        fprintf(stderr, "Allocation failed\n");
        sparse_free(A);
        free(b);
        free(x);
        free(ones);
        return 1;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* --- GMRES without preconditioner --- */
    printf("\n--- GMRES (no preconditioner) ---\n");
    sparse_gmres_opts_t opts = {.restart = 30, .max_iter = 500, .tol = 1e-10};
    sparse_iter_result_t result;
    memset(x, 0, (size_t)n * sizeof(double));

    sparse_err_t err = sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);
    printf("  Status:     %s\n", (err == SPARSE_OK) ? "converged" : "NOT converged");
    printf("  Iterations: %d\n", (int)result.iterations);
    printf("  Residual:   %.2e\n", result.residual_norm);

    /* Error vs exact solution */
    double err_norm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = x[i] - 1.0;
        err_norm += d * d;
    }
    printf("  ||x - x_exact||: %.2e\n", sqrt(err_norm));

    /* --- ILU(0) preconditioner --- */
    printf("\n--- GMRES with ILU(0) preconditioner ---\n");
    SparseMatrix *A_copy = sparse_copy(A);
    if (!A_copy) {
        fprintf(stderr, "Failed to copy for ILU\n");
        sparse_free(A);
        free(b);
        free(x);
        free(ones);
        return 1;
    }

    sparse_ilu_t ilu;
    err = sparse_ilu_factor(A_copy, &ilu);
    if (err != SPARSE_OK) {
        fprintf(stderr, "ILU factorization failed (err=%d)\n", (int)err);
        sparse_free(A);
        sparse_free(A_copy);
        free(b);
        free(x);
        free(ones);
        return 1;
    }
    printf("  ILU(0) factorized OK\n");

    /* GMRES with ILU preconditioner */
    sparse_gmres_opts_t opts_ilu = {.restart = 30, .max_iter = 500, .tol = 1e-10};
    memset(x, 0, (size_t)n * sizeof(double));

    err = sparse_solve_gmres(A, b, x, &opts_ilu, sparse_ilu_precond, &ilu, &result);
    printf("  Status:     %s\n", (err == SPARSE_OK) ? "converged" : "NOT converged");
    printf("  Iterations: %d\n", (int)result.iterations);
    printf("  Residual:   %.2e\n", result.residual_norm);

    err_norm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = x[i] - 1.0;
        err_norm += d * d;
    }
    printf("  ||x - x_exact||: %.2e\n", sqrt(err_norm));

    sparse_ilu_free(&ilu);
    sparse_free(A_copy);
    sparse_free(A);
    free(b);
    free(x);
    free(ones);
    return 0;
}
