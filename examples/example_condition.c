/*
 * example_condition.c — Condition number estimation and ill-conditioning.
 *
 * Demonstrates:
 *   - sparse_cond() for 2-norm condition number
 *   - How ill-conditioning amplifies errors in the solution
 *   - Comparing well-conditioned vs ill-conditioned systems
 *
 * Build:
 *   cc -O2 -Iinclude -o example_condition examples/example_condition.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_svd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Solve Ax = b and return ||x - x_exact|| */
static double solve_and_measure(SparseMatrix *A, const double *b, const double *x_exact, idx_t n) {
    SparseMatrix *LU = sparse_copy(A);
    if (!LU)
        return -1.0;

    sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-14);
    if (err != SPARSE_OK) {
        sparse_free(LU);
        return -1.0;
    }

    double *x = calloc((size_t)n, sizeof(double));
    if (!x) {
        sparse_free(LU);
        return -1.0;
    }

    err = sparse_lu_solve(LU, b, x);
    sparse_free(LU);
    if (err != SPARSE_OK) {
        free(x);
        return -1.0;
    }

    double err_norm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = x[i] - x_exact[i];
        err_norm += d * d;
    }
    free(x);
    return sqrt(err_norm);
}

int main(void) {
    printf("=== Condition Number and Ill-Conditioning Example ===\n\n");

    idx_t n = 4;

    /* --- Well-conditioned system: diagonal with ratio 10:1 --- */
    printf("--- Well-conditioned system (cond ≈ 10) ---\n");
    SparseMatrix *A_good = sparse_create(n, n);
    sparse_insert(A_good, 0, 0, 10.0);
    sparse_insert(A_good, 1, 1, 5.0);
    sparse_insert(A_good, 2, 2, 2.0);
    sparse_insert(A_good, 3, 3, 1.0);

    sparse_err_t cerr;
    double cond_good = sparse_cond(A_good, &cerr);
    printf("  Condition number: %.1f\n", cond_good);

    /* Exact solution: x = [1, 1, 1, 1] */
    double x_exact[] = {1.0, 1.0, 1.0, 1.0};
    double b_good[] = {10.0, 5.0, 2.0, 1.0};

    /* Perturb b slightly */
    double b_pert[4];
    double pert = 1e-10;
    for (idx_t i = 0; i < n; i++)
        b_pert[i] = b_good[i] + pert;

    double err_good = solve_and_measure(A_good, b_pert, x_exact, n);
    printf("  Perturbation in b: %.2e\n", pert * sqrt((double)n));
    printf("  Error in x:        %.2e\n", err_good);
    printf("  Error amplification: %.1fx\n\n", err_good / (pert * sqrt((double)n)));

    sparse_free(A_good);

    /* --- Ill-conditioned system: diagonal with ratio 1e8:1 --- */
    printf("--- Ill-conditioned system (cond ≈ 1e8) ---\n");
    SparseMatrix *A_bad = sparse_create(n, n);
    sparse_insert(A_bad, 0, 0, 1e4);
    sparse_insert(A_bad, 1, 1, 1.0);
    sparse_insert(A_bad, 2, 2, 1e-2);
    sparse_insert(A_bad, 3, 3, 1e-4);

    double cond_bad = sparse_cond(A_bad, &cerr);
    printf("  Condition number: %.2e\n", cond_bad);

    double b_bad[] = {1e4, 1.0, 1e-2, 1e-4};
    for (idx_t i = 0; i < n; i++)
        b_pert[i] = b_bad[i] + pert;

    double err_bad = solve_and_measure(A_bad, b_pert, x_exact, n);
    printf("  Perturbation in b: %.2e\n", pert * sqrt((double)n));
    printf("  Error in x:        %.2e\n", err_bad);
    printf("  Error amplification: %.1fx\n\n", err_bad / (pert * sqrt((double)n)));

    sparse_free(A_bad);

    /* --- Singular system: cond = infinity --- */
    printf("--- Singular system (cond = inf) ---\n");
    SparseMatrix *A_sing = sparse_create(3, 3);
    /* rank-2: row 2 = row 0 + row 1 */
    sparse_insert(A_sing, 0, 0, 1.0);
    sparse_insert(A_sing, 0, 1, 2.0);
    sparse_insert(A_sing, 1, 0, 3.0);
    sparse_insert(A_sing, 1, 1, 4.0);
    sparse_insert(A_sing, 2, 0, 4.0);
    sparse_insert(A_sing, 2, 1, 6.0);

    double cond_sing = sparse_cond(A_sing, &cerr);
    printf("  Condition number: %f\n", cond_sing);
    printf("  (Matrix is singular — no unique solution exists)\n");

    sparse_free(A_sing);

    printf("\nKey takeaway: the condition number bounds how much errors in b\n");
    printf("are amplified in the solution x. A rule of thumb: you lose\n");
    printf("log10(cond) digits of accuracy.\n");

    return 0;
}
