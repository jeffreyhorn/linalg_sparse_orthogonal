/*
 * example_ldlt.c — Solve a symmetric indefinite (KKT) system using LDL^T.
 *
 * Demonstrates:
 *   - Building a KKT saddle-point matrix K = [H A^T; A 0]
 *   - LDL^T factorization with Bunch-Kaufman pivoting
 *   - Solving K*x = b and checking the residual
 *   - Inertia computation (positive/negative eigenvalue counts)
 *   - Fill-reducing reordering with AMD
 *   - Iterative refinement
 *   - Condition number estimation
 *
 * Build:
 *   cc -O2 -Iinclude -o example_ldlt examples/example_ldlt.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_ldlt.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== LDL^T Factorization Example (KKT System) ===\n\n");

    /* Build a 6x6 KKT matrix:
     *
     *     [ H  A^T ]     H = [4 -1  0  0]    A = [1 0 1 0]
     * K = [        ]         [-1  4 -1  0]        [0 1 0 1]
     *     [ A   0  ]         [ 0 -1  4 -1]
     *                        [ 0  0 -1  4]
     *
     * K is 6x6, symmetric, indefinite (H is SPD but the zero block
     * introduces negative eigenvalues).  This structure arises in
     * constrained optimization (KKT conditions).
     */
    idx_t nh = 4; /* Hessian size */
    idx_t nc = 2; /* number of constraints */
    idx_t n = nh + nc;

    SparseMatrix *K = sparse_create(n, n);
    if (!K) {
        fprintf(stderr, "Failed to create matrix\n");
        return 1;
    }

    /* H block: tridiagonal SPD */
    for (idx_t i = 0; i < nh; i++) {
        sparse_insert(K, i, i, 4.0);
        if (i > 0) {
            sparse_insert(K, i, i - 1, -1.0);
            sparse_insert(K, i - 1, i, -1.0);
        }
    }

    /* A block and A^T block */
    sparse_insert(K, nh, 0, 1.0);
    sparse_insert(K, 0, nh, 1.0);
    sparse_insert(K, nh, 2, 1.0);
    sparse_insert(K, 2, nh, 1.0);
    sparse_insert(K, nh + 1, 1, 1.0);
    sparse_insert(K, 1, nh + 1, 1.0);
    sparse_insert(K, nh + 1, 3, 1.0);
    sparse_insert(K, 3, nh + 1, 1.0);

    printf("KKT matrix K (%d x %d, %d nonzeros):\n", (int)n, (int)n, (int)sparse_nnz(K));
    for (idx_t i = 0; i < n; i++) {
        printf("  [");
        for (idx_t j = 0; j < n; j++)
            printf(" %5.1f", sparse_get(K, i, j));
        printf(" ]\n");
    }

    /* ── Step 1: Factor without reordering ────────────────────────── */
    printf("\n--- Factorization (no reordering) ---\n");

    sparse_ldlt_t ldlt;
    sparse_err_t err = sparse_ldlt_factor(K, &ldlt);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Factorization failed: %s\n", sparse_strerror(err));
        sparse_free(K);
        return 1;
    }

    printf("  nnz(L) = %d\n", (int)sparse_nnz(ldlt.L));

    /* ── Step 2: Inertia ──────────────────────────────────────────── */
    idx_t pos, neg, zero;
    sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero);
    printf("  Inertia: (%d+, %d-, %d zero)\n", (int)pos, (int)neg, (int)zero);

    /* ── Step 3: Solve K*x = b ────────────────────────────────────── */
    printf("\n--- Solve ---\n");

    /* Known solution: x_exact = [1, 2, 3, 4, 0.5, -0.5] */
    double x_exact[] = {1.0, 2.0, 3.0, 4.0, 0.5, -0.5};
    double b[6], x[6], r[6];
    sparse_matvec(K, x_exact, b);

    printf("  b = [");
    for (int i = 0; i < (int)n; i++)
        printf(" %.4f", b[i]);
    printf(" ]\n");

    err = sparse_ldlt_solve(&ldlt, b, x);
    if (err != SPARSE_OK) {
        fprintf(stderr, "Solve failed: %s\n", sparse_strerror(err));
        sparse_ldlt_free(&ldlt);
        sparse_free(K);
        return 1;
    }

    printf("  x = [");
    for (int i = 0; i < (int)n; i++)
        printf(" %.4f", x[i]);
    printf(" ]\n");

    /* Residual check */
    sparse_matvec(K, x, r);
    double norm_res = 0.0, norm_b = 0.0;
    for (int i = 0; i < (int)n; i++) {
        double d = r[i] - b[i];
        norm_res += d * d;
        norm_b += b[i] * b[i];
    }
    printf("  Relative residual: %.3e\n", sqrt(norm_res / norm_b));

    sparse_ldlt_free(&ldlt);

    /* ── Step 4: Factor with AMD reordering ───────────────────────── */
    printf("\n--- Factorization with AMD reordering ---\n");

    sparse_ldlt_opts_t opts = {.reorder = SPARSE_REORDER_AMD};
    sparse_ldlt_t ldlt_amd;
    err = sparse_ldlt_factor_opts(K, &opts, &ldlt_amd);
    if (err != SPARSE_OK) {
        fprintf(stderr, "AMD factorization failed: %s\n", sparse_strerror(err));
        sparse_free(K);
        return 1;
    }

    printf("  nnz(L) = %d\n", (int)sparse_nnz(ldlt_amd.L));

    err = sparse_ldlt_solve(&ldlt_amd, b, x);
    if (err != SPARSE_OK) {
        fprintf(stderr, "AMD solve failed: %s\n", sparse_strerror(err));
        sparse_ldlt_free(&ldlt_amd);
        sparse_free(K);
        return 1;
    }

    /* ── Step 5: Iterative refinement ─────────────────────────────── */
    printf("\n--- Iterative refinement ---\n");

    sparse_matvec(K, x, r);
    norm_res = 0.0;
    for (int i = 0; i < (int)n; i++) {
        double d = r[i] - b[i];
        norm_res += d * d;
    }
    printf("  Before refinement: ||r|| = %.3e\n", sqrt(norm_res));

    sparse_ldlt_refine(K, &ldlt_amd, b, x, 5, 1e-15);

    sparse_matvec(K, x, r);
    norm_res = 0.0;
    for (int i = 0; i < (int)n; i++) {
        double d = r[i] - b[i];
        norm_res += d * d;
    }
    printf("  After refinement:  ||r|| = %.3e\n", sqrt(norm_res));

    /* ── Step 6: Condition estimation ─────────────────────────────── */
    printf("\n--- Condition estimation ---\n");

    double cond;
    sparse_ldlt_condest(K, &ldlt_amd, &cond);
    printf("  cond_1(K) ~ %.2f\n", cond);

    /* Cleanup */
    sparse_ldlt_free(&ldlt_amd);
    sparse_free(K);

    printf("\nDone.\n");
    return 0;
}
