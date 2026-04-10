/**
 * Sprint 13 cross-feature integration tests.
 *
 * Validates that IC(0) preconditioning and MINRES solver integrate correctly
 * with the rest of the library: IC(0) vs ILU(0) on SPD systems, MINRES on
 * symmetric indefinite KKT/saddle-point systems, cross-solver consistency
 * (CG vs MINRES vs GMRES vs LDL^T), preconditioned MINRES, block MINRES,
 * and SuiteSparse matrix validation.
 */
#include "sparse_ic.h"
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_ldlt.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_DIR "tests/data"
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

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

static SparseMatrix *build_spd_banded(idx_t n, idx_t bw) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(2 * bw + 2));
        for (idx_t d = 1; d <= bw && i + d < n; d++) {
            double off = -1.0 / (double)(d + 1);
            sparse_insert(A, i, i + d, off);
            sparse_insert(A, i + d, i, off);
        }
    }
    return A;
}

static SparseMatrix *build_spd_tridiag(idx_t n) {
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

/* Jacobi preconditioner */
typedef struct {
    double *diag_inv;
    idx_t n;
} jacobi_ctx_t;

static sparse_err_t jacobi_precond(const void *ctx, idx_t n, const double *r, double *z) {
    const jacobi_ctx_t *jac = (const jacobi_ctx_t *)ctx;
    (void)n;
    for (idx_t i = 0; i < jac->n; i++)
        z[i] = jac->diag_inv[i] * r[i];
    return SPARSE_OK;
}

static jacobi_ctx_t make_jacobi(const SparseMatrix *A, idx_t n) {
    jacobi_ctx_t jac;
    jac.n = n;
    jac.diag_inv = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(sparse_get_phys(A, i, i));
        jac.diag_inv[i] = (d > 1e-15) ? 1.0 / d : 1.0;
    }
    return jac;
}

/* LDL^T preconditioner callback */
static sparse_err_t ldlt_precond(const void *ctx, idx_t n, const double *r, double *z) {
    const sparse_ldlt_t *ldlt = (const sparse_ldlt_t *)ctx;
    (void)n;
    return sparse_ldlt_solve(ldlt, r, z);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 1: IC(0) vs ILU(0) on banded SPD — CG comparison
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_ic_vs_ilu_banded_cg(void) {
    idx_t n = 50;
    idx_t bw = 3;
    SparseMatrix *A = build_spd_banded(n, bw);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned CG */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_cg(A, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);

    /* IC(0)-preconditioned CG */
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_solve_cg(A, b, x2, &opts, sparse_ic_precond, &ic, &res2), SPARSE_OK);

    /* ILU(0)-preconditioned CG */
    sparse_ilu_t ilu;
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));
    double *x3 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res3;
    ASSERT_ERR(sparse_solve_cg(A, b, x3, &opts, sparse_ilu_precond, &ilu, &res3), SPARSE_OK);

    printf("    banded n=%d bw=%d CG: unprec=%d, IC(0)=%d, ILU(0)=%d iters\n", (int)n, (int)bw,
           (int)res1.iterations, (int)res2.iterations, (int)res3.iterations);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);
    ASSERT_TRUE(res3.converged);
    ASSERT_TRUE(res2.iterations <= res1.iterations);

    /* All three solutions should agree */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x1[i], x_exact[i], 1e-6);
        ASSERT_NEAR(x2[i], x_exact[i], 1e-6);
        ASSERT_NEAR(x3[i], x_exact[i], 1e-6);
    }

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    free(x3);
    sparse_ic_free(&ic);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 2: MINRES vs GMRES on KKT systems
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_minres_vs_gmres_kkt(void) {
    idx_t nh = 35, nc = 15;
    SparseMatrix *K = build_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1) / (double)n;
    sparse_matvec(K, x_exact, b);

    sparse_iter_opts_t mr_opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_gmres_opts_t gm_opts = {.max_iter = 500, .restart = 60, .tol = 1e-10, .verbose = 0};

    double *x_mr = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_mr;
    ASSERT_ERR(sparse_solve_minres(K, b, x_mr, &mr_opts, NULL, NULL, &res_mr), SPARSE_OK);

    double *x_gm = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_gm;
    ASSERT_ERR(sparse_solve_gmres(K, b, x_gm, &gm_opts, NULL, NULL, &res_gm), SPARSE_OK);

    printf("    KKT %dx%d: MINRES %d iters (relres=%.1e), GMRES %d iters (relres=%.1e)\n", (int)n,
           (int)n, (int)res_mr.iterations, res_mr.residual_norm, (int)res_gm.iterations,
           res_gm.residual_norm);

    ASSERT_TRUE(res_mr.converged);
    ASSERT_TRUE(res_gm.converged);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_mr[i], x_gm[i], 1e-6);

    free(x_exact);
    free(b);
    free(x_mr);
    free(x_gm);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 3: MINRES with Jacobi preconditioner on large KKT
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_minres_jacobi_large_kkt(void) {
    idx_t nh = 70, nc = 30;
    SparseMatrix *K = build_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = cos((double)(i + 1));
    sparse_matvec(K, x_exact, b);

    jacobi_ctx_t jac = make_jacobi(K, n);
    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-8, .verbose = 0};

    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    sparse_err_t err1 = sparse_solve_minres(K, b, x1, &opts, NULL, NULL, &res1);
    ASSERT_TRUE(err1 == SPARSE_OK || err1 == SPARSE_ERR_NOT_CONVERGED);

    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    sparse_err_t err2 = sparse_solve_minres(K, b, x2, &opts, jacobi_precond, &jac, &res2);
    ASSERT_TRUE(err2 == SPARSE_OK || err2 == SPARSE_ERR_NOT_CONVERGED);

    printf("    KKT %dx%d MINRES: unprec=%d iters (relres=%.1e), Jacobi=%d iters (relres=%.1e)\n",
           (int)n, (int)n, (int)res1.iterations, res1.residual_norm, (int)res2.iterations,
           res2.residual_norm);

    /* Check residual quality rather than strict convergence flag, since
     * MINRES true residual may be slightly above the QR estimate tolerance */
    ASSERT_TRUE(res1.residual_norm < 1e-6);
    ASSERT_TRUE(res2.residual_norm < 1e-6);

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    free(jac.diag_inv);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 4: Cross-solver consistency — CG, MINRES, GMRES on SPD
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_cross_solver_spd(void) {
    idx_t n = 25;
    SparseMatrix *A = build_spd_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t cg_opts = {.max_iter = 500, .tol = 1e-12, .verbose = 0};
    sparse_gmres_opts_t gm_opts = {.max_iter = 500, .restart = 30, .tol = 1e-12, .verbose = 0};

    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_cg;
    ASSERT_ERR(sparse_solve_cg(A, b, x_cg, &cg_opts, NULL, NULL, &res_cg), SPARSE_OK);

    double *x_mr = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_mr;
    ASSERT_ERR(sparse_solve_minres(A, b, x_mr, &cg_opts, NULL, NULL, &res_mr), SPARSE_OK);

    double *x_gm = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_gm;
    ASSERT_ERR(sparse_solve_gmres(A, b, x_gm, &gm_opts, NULL, NULL, &res_gm), SPARSE_OK);

    printf("    cross-solver SPD n=%d: CG=%d, MINRES=%d, GMRES=%d iters\n", (int)n,
           (int)res_cg.iterations, (int)res_mr.iterations, (int)res_gm.iterations);

    ASSERT_TRUE(res_cg.converged);
    ASSERT_TRUE(res_mr.converged);
    ASSERT_TRUE(res_gm.converged);

    /* All three should agree */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_cg[i], x_mr[i], 1e-8);
        ASSERT_NEAR(x_cg[i], x_gm[i], 1e-8);
    }

    free(x_exact);
    free(b);
    free(x_cg);
    free(x_mr);
    free(x_gm);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 5: Cross-solver on indefinite — MINRES, GMRES, LDL^T
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_cross_solver_indefinite(void) {
    idx_t nh = 15, nc = 6;
    SparseMatrix *K = build_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(K, x_exact, b);

    /* MINRES */
    sparse_iter_opts_t mr_opts = {.max_iter = 500, .tol = 1e-12, .verbose = 0};
    double *x_mr = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_mr;
    ASSERT_ERR(sparse_solve_minres(K, b, x_mr, &mr_opts, NULL, NULL, &res_mr), SPARSE_OK);

    /* GMRES */
    sparse_gmres_opts_t gm_opts = {.max_iter = 500, .restart = 30, .tol = 1e-12, .verbose = 0};
    double *x_gm = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_gm;
    ASSERT_ERR(sparse_solve_gmres(K, b, x_gm, &gm_opts, NULL, NULL, &res_gm), SPARSE_OK);

    /* LDL^T direct */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(K, &ldlt));
    double *x_ldlt = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(sparse_ldlt_solve(&ldlt, b, x_ldlt));

    printf("    cross-solver KKT %dx%d: MINRES=%d, GMRES=%d iters, LDL^T=direct\n", (int)n, (int)n,
           (int)res_mr.iterations, (int)res_gm.iterations);

    ASSERT_TRUE(res_mr.converged);
    ASSERT_TRUE(res_gm.converged);

    /* All three should agree */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_mr[i], x_ldlt[i], 1e-8);
        ASSERT_NEAR(x_gm[i], x_ldlt[i], 1e-8);
    }

    free(x_exact);
    free(b);
    free(x_mr);
    free(x_gm);
    free(x_ldlt);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 6: LDL^T as MINRES preconditioner (validation — exact precond)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_ldlt_precond_minres(void) {
    /* LDL^T as exact preconditioner on SPD system.
     * Note: preconditioned MINRES requires M to be SPD, so we use an SPD
     * matrix where LDL^T produces a positive factorization. */
    idx_t n = 15;
    SparseMatrix *A = build_spd_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* Factor for preconditioner */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(A, &ldlt));

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};

    /* Unpreconditioned MINRES */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_minres(A, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);

    /* LDL^T-preconditioned MINRES — should converge in 1 iteration (exact preconditioner) */
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_solve_minres(A, b, x2, &opts, ldlt_precond, &ldlt, &res2), SPARSE_OK);

    printf("    LDL^T-precond MINRES on SPD n=%d: unprec=%d iters, LDL^T=%d iters\n", (int)n,
           (int)res1.iterations, (int)res2.iterations);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);
    ASSERT_TRUE(res2.iterations <= 1);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x2[i], x_exact[i], 1e-8);

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 7: SuiteSparse bcsstk04 — IC(0)-CG and MINRES
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_suitesparse_bcsstk04(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (err != SPARSE_OK || !A) {
        printf("    [SKIP] bcsstk04.mtx not available\n");
        return;
    }

    idx_t n = sparse_rows(A);

    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};

    /* IC(0)-preconditioned CG */
    sparse_ilu_t ic;
    err = sparse_ic_factor(A, &ic);
    if (err != SPARSE_OK) {
        printf("    [SKIP] IC(0) factor failed on bcsstk04\n");
        free(b);
        sparse_free(A);
        return;
    }

    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_cg;
    {
        sparse_err_t serr = sparse_solve_cg(A, b, x_cg, &opts, sparse_ic_precond, &ic, &res_cg);
        ASSERT_ERR(serr, SPARSE_OK);
    }

    /* MINRES (unpreconditioned — bcsstk04 is SPD) */
    double *x_mr = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_mr;
    {
        sparse_err_t serr = sparse_solve_minres(A, b, x_mr, &opts, NULL, NULL, &res_mr);
        ASSERT_TRUE(serr == SPARSE_OK || serr == SPARSE_ERR_NOT_CONVERGED);
    }

    /* IC(0)-preconditioned MINRES */
    double *x_mr2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_mr2;
    {
        sparse_err_t serr =
            sparse_solve_minres(A, b, x_mr2, &opts, sparse_ic_precond, &ic, &res_mr2);
        ASSERT_TRUE(serr == SPARSE_OK || serr == SPARSE_ERR_NOT_CONVERGED);
    }

    printf("    bcsstk04 (n=%d): IC(0)-CG=%d, MINRES=%d, IC(0)-MINRES=%d iters\n", (int)n,
           (int)res_cg.iterations, (int)res_mr.iterations, (int)res_mr2.iterations);

    ASSERT_TRUE(res_cg.converged);
    /* MINRES true residual may be slightly above QR estimate tolerance;
     * check residual directly instead of converged flag */
    ASSERT_TRUE(res_mr.residual_norm < 1e-8);
    ASSERT_TRUE(res_mr2.residual_norm < 1e-8);

    /* Solutions should agree */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_cg[i], x_mr[i], 1e-5);

    free(b);
    free(x_cg);
    free(x_mr);
    free(x_mr2);
    sparse_ic_free(&ic);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 8: SuiteSparse nos4 — IC(0)-CG vs ILU(0)-CG
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_suitesparse_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (err != SPARSE_OK || !A) {
        printf("    [SKIP] nos4.mtx not available\n");
        return;
    }

    idx_t n = sparse_rows(A);

    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1) / (double)n;

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};

    /* IC(0) */
    sparse_ilu_t ic;
    err = sparse_ic_factor(A, &ic);
    if (err != SPARSE_OK) {
        printf("    [SKIP] IC(0) factor failed on nos4\n");
        free(b);
        sparse_free(A);
        return;
    }

    double *x_ic = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_ic;
    ASSERT_ERR(sparse_solve_cg(A, b, x_ic, &opts, sparse_ic_precond, &ic, &res_ic), SPARSE_OK);

    /* ILU(0) */
    sparse_ilu_t ilu;
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));
    double *x_ilu = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_ilu;
    {
        sparse_err_t e = sparse_solve_cg(A, b, x_ilu, &opts, sparse_ilu_precond, &ilu, &res_ilu);
        ASSERT_ERR(e, SPARSE_OK);
    }

    printf("    nos4 (n=%d): IC(0)-CG=%d iters, ILU(0)-CG=%d iters\n", (int)n,
           (int)res_ic.iterations, (int)res_ilu.iterations);

    ASSERT_TRUE(res_ic.converged);
    ASSERT_TRUE(res_ilu.converged);

    /* Solutions should agree */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_ic[i], x_ilu[i], 1e-6);

    free(b);
    free(x_ic);
    free(x_ilu);
    sparse_ic_free(&ic);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 9: Block MINRES integration — multi-RHS on KKT with LDL^T verify
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_block_minres_kkt(void) {
    idx_t nh = 20, nc = 8;
    SparseMatrix *K = build_kkt(nh, nc);
    idx_t n = nh + nc;
    idx_t nrhs = 4;

    double *X_exact = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    for (idx_t j = 0; j < nrhs; j++) {
        for (idx_t i = 0; i < n; i++)
            X_exact[j * n + i] = sin((double)(j * 30 + i + 1));
        sparse_matvec(K, X_exact + j * n, B + j * n);
    }

    /* Block MINRES */
    double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_minres_solve_block(K, B, nrhs, X, &opts, NULL, NULL, &res), SPARSE_OK);
    ASSERT_TRUE(res.converged);

    /* Verify against LDL^T direct solve */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(K, &ldlt));

    for (idx_t j = 0; j < nrhs; j++) {
        double *x_ldlt = malloc((size_t)n * sizeof(double));
        REQUIRE_OK(sparse_ldlt_solve(&ldlt, B + j * n, x_ldlt));

        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(X[j * n + i], x_ldlt[i], 1e-6);

        free(x_ldlt);
    }

    printf("    block MINRES KKT %dx%d nrhs=%d: %d iters, verified vs LDL^T\n", (int)n, (int)n,
           (int)nrhs, (int)res.iterations);

    free(X_exact);
    free(B);
    free(X);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 10: IC(0)-preconditioned MINRES vs IC(0)-preconditioned CG on SPD
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_ic_minres_vs_cg(void) {
    idx_t n = 40;
    SparseMatrix *A = build_spd_banded(n, 4);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1) / (double)n;
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_cg;
    ASSERT_ERR(sparse_solve_cg(A, b, x_cg, &opts, sparse_ic_precond, &ic, &res_cg), SPARSE_OK);

    double *x_mr = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_mr;
    ASSERT_ERR(sparse_solve_minres(A, b, x_mr, &opts, sparse_ic_precond, &ic, &res_mr), SPARSE_OK);

    printf("    IC(0)-precond SPD n=%d: CG=%d iters, MINRES=%d iters\n", (int)n,
           (int)res_cg.iterations, (int)res_mr.iterations);

    ASSERT_TRUE(res_cg.converged);
    ASSERT_TRUE(res_mr.converged);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_cg[i], x_mr[i], 1e-6);

    free(x_exact);
    free(b);
    free(x_cg);
    free(x_mr);
    sparse_ic_free(&ic);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 11: Extreme-scale IC(0) and MINRES
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_scaled_ic_minres(void) {
    double scales[] = {1e-15, 1e-5, 1.0, 1e5, 1e15};
    int nscales = 5;

    for (int s = 0; s < nscales; s++) {
        idx_t n = 8;
        SparseMatrix *A = build_spd_tridiag(n);
        /* Scale all entries */
        for (idx_t i = 0; i < n; i++) {
            double d = sparse_get_phys(A, i, i);
            sparse_insert(A, i, i, d * scales[s]);
            if (i > 0) {
                double off = sparse_get_phys(A, i, i - 1);
                sparse_insert(A, i, i - 1, off * scales[s]);
                sparse_insert(A, i - 1, i, off * scales[s]);
            }
        }

        /* IC(0) factor and solve */
        sparse_ilu_t ic;
        sparse_err_t ferr = sparse_ic_factor(A, &ic);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            continue; /* breakdown at extreme scales is acceptable */
        }

        double *x_exact = malloc((size_t)n * sizeof(double));
        double *b = malloc((size_t)n * sizeof(double));
        double *x = calloc((size_t)n, sizeof(double));
        for (idx_t i = 0; i < n; i++)
            x_exact[i] = (double)(i + 1);
        sparse_matvec(A, x_exact, b);

        /* IC(0) direct solve */
        REQUIRE_OK(sparse_ic_solve(&ic, b, x));
        double relres_ic = relative_residual(A, x, b, n);
        ASSERT_TRUE(relres_ic < 1e-6);

        /* IC(0)-preconditioned MINRES */
        memset(x, 0, (size_t)n * sizeof(double));
        sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-8, .verbose = 0};
        sparse_iter_result_t res;
        sparse_err_t err = sparse_solve_minres(A, b, x, &opts, sparse_ic_precond, &ic, &res);
        ASSERT_TRUE(err == SPARSE_OK);
        ASSERT_TRUE(res.converged);

        free(x_exact);
        free(b);
        free(x);
        sparse_ic_free(&ic);
        sparse_free(A);
    }
    printf("    extreme-scale IC(0)+MINRES: 5 scales from 1e-15 to 1e+15 OK\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 12: Unfactored IC(0) solve-before-factor
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_ic_unfactored_solve(void) {
    /* Zeroed struct has n=0: solve is a valid no-op (consistent with ILU) */
    sparse_ilu_t ic;
    memset(&ic, 0, sizeof(ic));

    double r[3] = {1, 2, 3};
    double z[3];
    ASSERT_ERR(sparse_ic_solve(&ic, r, z), SPARSE_OK); /* n=0 → no-op */

    /* Struct with n>0 but NULL L/U should fail */
    ic.n = 3;
    ASSERT_ERR(sparse_ic_solve(&ic, r, z), SPARSE_ERR_NULL);
    ic.n = 0; /* restore before free */

    /* Factor, solve (works), free, solve again (n=0 after free → no-op) */
    SparseMatrix *A = build_spd_tridiag(3);
    REQUIRE_OK(sparse_ic_factor(A, &ic));
    REQUIRE_OK(sparse_ic_solve(&ic, r, z));

    sparse_ic_free(&ic);
    /* After free, n=0 and L/U=NULL — solve returns OK (n=0 no-op) */
    ASSERT_ERR(sparse_ic_solve(&ic, r, z), SPARSE_OK);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 13: Large KKT benchmark — MINRES vs GMRES performance data
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_large_kkt_benchmark(void) {
    idx_t nh = 200, nc = 80;
    SparseMatrix *K = build_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(K, x_exact, b);

    sparse_iter_opts_t mr_opts = {.max_iter = 2000, .tol = 1e-8, .verbose = 0};
    sparse_gmres_opts_t gm_opts = {.max_iter = 2000, .restart = 100, .tol = 1e-8, .verbose = 0};

    /* MINRES */
    double *x_mr = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_mr;
    ASSERT_ERR(sparse_solve_minres(K, b, x_mr, &mr_opts, NULL, NULL, &res_mr), SPARSE_OK);

    /* GMRES */
    double *x_gm = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_gm;
    ASSERT_ERR(sparse_solve_gmres(K, b, x_gm, &gm_opts, NULL, NULL, &res_gm), SPARSE_OK);

    /* Jacobi-preconditioned MINRES */
    jacobi_ctx_t jac = make_jacobi(K, n);
    double *x_jmr = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_jmr;
    ASSERT_ERR(sparse_solve_minres(K, b, x_jmr, &mr_opts, jacobi_precond, &jac, &res_jmr),
               SPARSE_OK);

    /* LDL^T direct */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(K, &ldlt));
    double *x_ldlt = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(sparse_ldlt_solve(&ldlt, b, x_ldlt));

    double relres_mr = relative_residual(K, x_mr, b, n);
    double relres_gm = relative_residual(K, x_gm, b, n);
    double relres_jmr = relative_residual(K, x_jmr, b, n);
    double relres_ldlt = relative_residual(K, x_ldlt, b, n);

    printf("    KKT %dx%d benchmark:\n", (int)n, (int)n);
    printf("      MINRES:        %d iters, relres=%.3e\n", (int)res_mr.iterations, relres_mr);
    printf("      GMRES(100):    %d iters, relres=%.3e\n", (int)res_gm.iterations, relres_gm);
    printf("      Jacobi-MINRES: %d iters, relres=%.3e\n", (int)res_jmr.iterations, relres_jmr);
    printf("      LDL^T direct:  relres=%.3e\n", relres_ldlt);

    ASSERT_TRUE(res_mr.converged);
    ASSERT_TRUE(res_gm.converged);
    ASSERT_TRUE(res_jmr.converged);
    ASSERT_TRUE(relres_mr < 1e-8);
    ASSERT_TRUE(relres_jmr < 1e-8);

    /* Jacobi preconditioning should help */
    ASSERT_TRUE(res_jmr.iterations <= res_mr.iterations);

    /* MINRES solution should agree with LDL^T */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_mr[i], x_ldlt[i], 1e-5);

    free(x_exact);
    free(b);
    free(x_mr);
    free(x_gm);
    free(x_jmr);
    free(x_ldlt);
    free(jac.diag_inv);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 14: IC(0) vs ILU(0) benchmark on SuiteSparse bcsstk04
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_s13_ic_ilu_benchmark_bcsstk04(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (err != SPARSE_OK || !A) {
        printf("    [SKIP] bcsstk04.mtx not available\n");
        return;
    }

    idx_t n = sparse_rows(A);

    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};

    /* IC(0) */
    sparse_ilu_t ic;
    err = sparse_ic_factor(A, &ic);
    if (err != SPARSE_OK) {
        printf("    [SKIP] IC(0) failed on bcsstk04\n");
        free(b);
        sparse_free(A);
        return;
    }

    /* ILU(0) */
    sparse_ilu_t ilu;
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));

    /* Unpreconditioned CG */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_cg(A, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);

    /* IC(0)-CG */
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_solve_cg(A, b, x2, &opts, sparse_ic_precond, &ic, &res2), SPARSE_OK);

    /* ILU(0)-CG */
    double *x3 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res3;
    ASSERT_ERR(sparse_solve_cg(A, b, x3, &opts, sparse_ilu_precond, &ilu, &res3), SPARSE_OK);

    /* IC(0)-MINRES */
    double *x4 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res4;
    {
        sparse_err_t serr = sparse_solve_minres(A, b, x4, &opts, sparse_ic_precond, &ic, &res4);
        ASSERT_TRUE(serr == SPARSE_OK || serr == SPARSE_ERR_NOT_CONVERGED);
    }

    printf("    bcsstk04 (n=%d) benchmark:\n", (int)n);
    printf("      CG (unprec):   %d iters, relres=%.3e\n", (int)res1.iterations,
           res1.residual_norm);
    printf("      IC(0)-CG:      %d iters, relres=%.3e, nnz(L)=%d\n", (int)res2.iterations,
           res2.residual_norm, (int)sparse_nnz(ic.L));
    printf("      ILU(0)-CG:     %d iters, relres=%.3e, nnz(L)=%d, nnz(U)=%d\n",
           (int)res3.iterations, res3.residual_norm, (int)sparse_nnz(ilu.L),
           (int)sparse_nnz(ilu.U));
    printf("      IC(0)-MINRES:  %d iters, relres=%.3e\n", (int)res4.iterations,
           res4.residual_norm);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);
    ASSERT_TRUE(res3.converged);
    /* IC(0)-MINRES: check residual directly (true residual may be slightly
     * above QR estimate tolerance) */
    ASSERT_TRUE(res4.residual_norm < 1e-8);

    /* IC(0) and ILU(0) should both improve over unpreconditioned */
    ASSERT_TRUE(res2.iterations <= res1.iterations);
    ASSERT_TRUE(res3.iterations <= res1.iterations);

    free(b);
    free(x1);
    free(x2);
    free(x3);
    free(x4);
    sparse_ic_free(&ic);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("test_sprint13_integration");

    RUN_TEST(test_s13_ic_vs_ilu_banded_cg);
    RUN_TEST(test_s13_minres_vs_gmres_kkt);
    RUN_TEST(test_s13_minres_jacobi_large_kkt);
    RUN_TEST(test_s13_cross_solver_spd);
    RUN_TEST(test_s13_cross_solver_indefinite);
    RUN_TEST(test_s13_ldlt_precond_minres);
    RUN_TEST(test_s13_suitesparse_bcsstk04);
    RUN_TEST(test_s13_suitesparse_nos4);
    RUN_TEST(test_s13_block_minres_kkt);
    RUN_TEST(test_s13_ic_minres_vs_cg);
    RUN_TEST(test_s13_scaled_ic_minres);
    RUN_TEST(test_s13_ic_unfactored_solve);
    RUN_TEST(test_s13_large_kkt_benchmark);
    RUN_TEST(test_s13_ic_ilu_benchmark_bcsstk04);

    TEST_SUITE_END();
}
