/*
 * example_analysis.c — Symbolic analysis with numeric refactorization.
 *
 * Demonstrates the analyze-once, factor-many workflow:
 *   1. Build a sparse SPD matrix
 *   2. Compute symbolic analysis once (etree, column counts, symbolic structure)
 *   3. Perform numeric Cholesky factorization
 *   4. Solve a linear system
 *   5. Modify matrix values (same sparsity pattern)
 *   6. Refactor numerically (no re-analysis)
 *   7. Solve the updated system
 *
 * This pattern is critical for nonlinear solvers, time-stepping codes,
 * and optimization loops that solve many systems with the same structure
 * but different values.
 *
 * Build:
 *   cc -O2 -Iinclude -o example_analysis examples/example_analysis.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_analysis.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Build an n×n SPD tridiagonal matrix with parameterized diagonal */
static SparseMatrix *make_tridiag(idx_t n, double diag_val) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, diag_val);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

/* Compute ||b - A*x||_inf / ||b||_inf */
static double residual(const SparseMatrix *A, const double *b, const double *x, idx_t n) {
    double *Ax = calloc((size_t)n, sizeof(double));
    sparse_matvec(A, x, Ax);
    double rnorm = 0, bnorm = 0;
    for (idx_t i = 0; i < n; i++) {
        double ri = fabs(b[i] - Ax[i]);
        if (ri > rnorm)
            rnorm = ri;
        if (fabs(b[i]) > bnorm)
            bnorm = fabs(b[i]);
    }
    free(Ax);
    return bnorm > 0 ? rnorm / bnorm : rnorm;
}

int main(void) {
    printf("=== Symbolic Analysis + Numeric Refactorization Example ===\n\n");

    idx_t n = 200;
    int num_refactors = 20;

    /* ── Step 1: Build the initial matrix ──────────────────────────── */
    SparseMatrix *A = make_tridiag(n, 4.0);
    if (!A) {
        fprintf(stderr, "Failed to create matrix\n");
        return 1;
    }
    printf("Matrix: %d×%d tridiagonal SPD, nnz = %d\n\n", (int)n, (int)n, (int)sparse_nnz(A));

    /* ── Step 2: Symbolic analysis (done once) ─────────────────────── */
    sparse_analysis_opts_t opts = {SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_NONE};
    sparse_analysis_t analysis = {0};

    clock_t t0 = clock();
    sparse_err_t err = sparse_analyze(A, &opts, &analysis);
    clock_t t1 = clock();

    if (err != SPARSE_OK) {
        fprintf(stderr, "Analysis failed: %s\n", sparse_strerror(err));
        sparse_free(A);
        return 1;
    }

    double analysis_time = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Symbolic analysis:\n");
    printf("  Predicted nnz(L) = %d\n", (int)analysis.sym_L.nnz);
    printf("  ||A||_inf        = %.1f\n", analysis.analysis_norm);
    printf("  Time             = %.6f s\n\n", analysis_time);

    /* ── Step 3: Numeric factorization ─────────────────────────────── */
    sparse_factors_t factors = {0};

    t0 = clock();
    err = sparse_factor_numeric(A, &analysis, &factors);
    t1 = clock();

    if (err != SPARSE_OK) {
        fprintf(stderr, "Factorization failed: %s\n", sparse_strerror(err));
        sparse_analysis_free(&analysis);
        sparse_free(A);
        return 1;
    }

    double factor_time = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Initial numeric factorization: %.6f s\n", factor_time);

    /* ── Step 4: Solve ─────────────────────────────────────────────── */
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    err = sparse_factor_solve(&factors, &analysis, b, x);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Solve failed: %s\n", sparse_strerror(err));
    } else {
        printf("  Solve residual: %.2e\n\n", residual(A, b, x, n));
    }

    /* ── Steps 5-7: Repeated refactorization ──────────────────────── */
    printf("Refactoring %d times with varying diagonal...\n", num_refactors);

    double total_refactor_time = 0;
    for (int iter = 0; iter < num_refactors; iter++) {
        /* Build new matrix with same pattern, different diagonal */
        double new_diag = 4.0 + 0.5 * (double)iter;
        SparseMatrix *A_new = make_tridiag(n, new_diag);
        if (!A_new)
            break;

        t0 = clock();
        err = sparse_refactor_numeric(A_new, &analysis, &factors);
        t1 = clock();
        total_refactor_time += (double)(t1 - t0) / CLOCKS_PER_SEC;

        if (err != SPARSE_OK) {
            fprintf(stderr, "  Refactor %d failed: %s\n", iter, sparse_strerror(err));
            sparse_free(A_new);
            break;
        }

        err = sparse_factor_solve(&factors, &analysis, b, x);
        if (err != SPARSE_OK) {
            fprintf(stderr, "  Solve %d failed: %s\n", iter, sparse_strerror(err));
            sparse_free(A_new);
            break;
        }

        double resid = residual(A_new, b, x, n);
        if (iter == 0 || iter == num_refactors - 1) {
            printf("  iter %2d: diag=%.1f, residual=%.2e\n", iter, new_diag, resid);
        }

        sparse_free(A_new);
    }

    printf("\nTiming summary:\n");
    printf("  Analysis (once):           %.6f s\n", analysis_time);
    printf("  Initial factorization:     %.6f s\n", factor_time);
    printf("  %d refactorizations:       %.6f s (avg %.6f s each)\n", num_refactors,
           total_refactor_time, total_refactor_time / num_refactors);
    printf("  Savings: analysis cost amortized over %d solves\n\n", num_refactors + 1);

    /* ── Cleanup ──────────────────────────────────────────────────── */
    free(b);
    free(x);
    sparse_factor_free(&factors);
    sparse_analysis_free(&analysis);
    sparse_free(A);

    printf("Done.\n");
    return 0;
}
