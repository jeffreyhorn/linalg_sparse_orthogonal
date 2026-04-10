/*
 * example_ic_minres.c — IC(0) preconditioning and MINRES solver demo.
 *
 * Demonstrates:
 *   - IC(0) factorization of an SPD matrix
 *   - IC(0) as a CG preconditioner (iteration count reduction)
 *   - MINRES on a symmetric indefinite (KKT) system
 *   - Preconditioned MINRES with a diagonal (Jacobi) preconditioner
 *   - Block MINRES for multiple right-hand sides
 *
 * Build:
 *   cc -O2 -Iinclude -o example_ic_minres examples/example_ic_minres.c \
 *      -Lbuild -lsparse_lu_ortho -lm
 */
#include "sparse_ic.h"
#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build an n x n SPD tridiagonal matrix: diag=4, off=-1 */
static SparseMatrix *build_spd(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

/* Build a KKT matrix K = [H A^T; A 0] */
static SparseMatrix *build_kkt(idx_t nh, idx_t nc) {
    idx_t n = nh + nc;
    SparseMatrix *K = sparse_create(n, n);
    for (idx_t i = 0; i < nh; i++) {
        sparse_insert(K, i, i, 4.0);
        if (i > 0) {
            sparse_insert(K, i, i - 1, -1.0);
            sparse_insert(K, i - 1, i, -1.0);
        }
    }
    for (idx_t c = 0; c < nc; c++) {
        idx_t j0 = (c * 2) % nh;
        idx_t j1 = (j0 + 1) % nh;
        sparse_insert(K, nh + c, j0, 1.0);
        sparse_insert(K, j0, nh + c, 1.0);
        sparse_insert(K, nh + c, j1, 1.0);
        sparse_insert(K, j1, nh + c, 1.0);
    }
    return K;
}

/* Jacobi preconditioner */
static double *jac_diag_inv;
static idx_t jac_n;

static sparse_err_t jacobi_apply(const void *ctx, idx_t n, const double *r, double *z) {
    (void)ctx;
    (void)n;
    for (idx_t i = 0; i < jac_n; i++)
        z[i] = jac_diag_inv[i] * r[i];
    return SPARSE_OK;
}

static double relative_residual(const SparseMatrix *A, const double *x, const double *b, idx_t n) {
    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(A, x, r);
    double nr = 0.0, nb = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = r[i] - b[i];
        nr += d * d;
        nb += b[i] * b[i];
    }
    free(r);
    return (nb > 0.0) ? sqrt(nr / nb) : sqrt(nr);
}

int main(void) {
    printf("=== IC(0) Preconditioning & MINRES Demo ===\n\n");

    /* ── Part 1: IC(0)-preconditioned CG on SPD system ──────────── */
    printf("--- Part 1: IC(0) as CG preconditioner ---\n");
    {
        idx_t n = 50;
        SparseMatrix *A = build_spd(n);

        double *x_exact = malloc((size_t)n * sizeof(double));
        double *b = malloc((size_t)n * sizeof(double));
        for (idx_t i = 0; i < n; i++)
            x_exact[i] = (double)(i + 1) / (double)n;
        sparse_matvec(A, x_exact, b);

        sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

        /* Unpreconditioned CG */
        double *x1 = calloc((size_t)n, sizeof(double));
        sparse_iter_result_t res1;
        sparse_solve_cg(A, b, x1, &opts, NULL, NULL, &res1);
        printf("  Unpreconditioned CG: %d iters, relres=%.2e\n", (int)res1.iterations,
               res1.residual_norm);

        /* IC(0)-preconditioned CG */
        sparse_ilu_t ic;
        sparse_ic_factor(A, &ic);

        double *x2 = calloc((size_t)n, sizeof(double));
        sparse_iter_result_t res2;
        sparse_solve_cg(A, b, x2, &opts, sparse_ic_precond, &ic, &res2);
        printf("  IC(0)-CG:            %d iters, relres=%.2e  (%.1fx speedup)\n",
               (int)res2.iterations, res2.residual_norm,
               (double)res1.iterations / (double)(res2.iterations > 0 ? res2.iterations : 1));

        free(x_exact);
        free(b);
        free(x1);
        free(x2);
        sparse_ic_free(&ic);
        sparse_free(A);
    }

    /* ── Part 2: MINRES on symmetric indefinite KKT system ──────── */
    printf("\n--- Part 2: MINRES on symmetric indefinite system ---\n");
    {
        idx_t nh = 30, nc = 12;
        SparseMatrix *K = build_kkt(nh, nc);
        idx_t n = nh + nc;

        double *x_exact = malloc((size_t)n * sizeof(double));
        double *b = malloc((size_t)n * sizeof(double));
        for (idx_t i = 0; i < n; i++)
            x_exact[i] = sin((double)(i + 1));
        sparse_matvec(K, x_exact, b);

        sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

        /* Unpreconditioned MINRES */
        double *x1 = calloc((size_t)n, sizeof(double));
        sparse_iter_result_t res1;
        sparse_solve_minres(K, b, x1, &opts, NULL, NULL, &res1);

        double relres = relative_residual(K, x1, b, n);
        printf("  KKT system %dx%d (symmetric indefinite)\n", (int)n, (int)n);
        printf("  MINRES:       %d iters, relres=%.2e\n", (int)res1.iterations, relres);

        /* Jacobi-preconditioned MINRES */
        jac_n = n;
        jac_diag_inv = malloc((size_t)n * sizeof(double));
        for (idx_t i = 0; i < n; i++) {
            double d = fabs(sparse_get_phys(K, i, i));
            jac_diag_inv[i] = (d > 1e-15) ? 1.0 / d : 1.0;
        }

        double *x2 = calloc((size_t)n, sizeof(double));
        sparse_iter_result_t res2;
        sparse_solve_minres(K, b, x2, &opts, jacobi_apply, NULL, &res2);

        double relres2 = relative_residual(K, x2, b, n);
        printf("  Jacobi-MINRES: %d iters, relres=%.2e  (%.1fx speedup)\n", (int)res2.iterations,
               relres2,
               (double)res1.iterations / (double)(res2.iterations > 0 ? res2.iterations : 1));

        free(x_exact);
        free(b);
        free(x1);
        free(x2);
        free(jac_diag_inv);
        sparse_free(K);
    }

    /* ── Part 3: Block MINRES — multiple RHS ────────────────────── */
    printf("\n--- Part 3: Block MINRES (multiple RHS) ---\n");
    {
        idx_t nh = 20, nc = 8;
        SparseMatrix *K = build_kkt(nh, nc);
        idx_t n = nh + nc;
        idx_t nrhs = 3;

        double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
        for (idx_t j = 0; j < nrhs; j++)
            for (idx_t i = 0; i < n; i++)
                B[j * n + i] = sin((double)(j * 100 + i + 1));

        double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
        sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
        sparse_iter_result_t res;
        sparse_minres_solve_block(K, B, nrhs, X, &opts, NULL, NULL, &res);

        printf("  KKT %dx%d, %d RHS: %d max iters, relres=%.2e, converged=%s\n", (int)n, (int)n,
               (int)nrhs, (int)res.iterations, res.residual_norm, res.converged ? "yes" : "no");

        free(B);
        free(X);
        sparse_free(K);
    }

    printf("\nDone.\n");
    return 0;
}
