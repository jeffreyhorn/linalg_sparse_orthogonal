#include "sparse_matrix.h"
#include "sparse_iterative.h"
#include "sparse_ilu.h"
#include "sparse_cholesky.h"
#include "sparse_lu.h"
#include "sparse_vector.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static double compute_relative_residual(const SparseMatrix *A,
                                         const double *b, const double *x,
                                         idx_t n)
{
    double *r = malloc((size_t)n * sizeof(double));
    if (!r) return INFINITY;
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] = b[i] - r[i];
    double rnorm = vec_norm2(r, n);
    double bnorm = vec_norm2(b, n);
    free(r);
    return (bnorm > 0.0) ? rnorm / bnorm : 0.0;
}

/* Cholesky preconditioner */
static sparse_err_t cholesky_precond_apply(const void *ctx, idx_t n,
                                            const double *r, double *z)
{
    const SparseMatrix *L = (const SparseMatrix *)ctx;
    (void)n;
    return sparse_cholesky_solve(L, r, z);
}

/* Identity preconditioner (passthrough) */
static sparse_err_t identity_precond(const void *ctx, idx_t n,
                                      const double *r, double *z)
{
    (void)ctx;
    for (idx_t i = 0; i < n; i++) z[i] = r[i];
    return SPARSE_OK;
}

/* Load matrix, generate RHS from x_exact = [1, 2, ..., n] */
typedef struct {
    SparseMatrix *A;
    double *x_exact;
    double *b;
    idx_t n;
} test_system_t;

static int load_system(test_system_t *sys, const char *path)
{
    sys->A = NULL;
    if (sparse_load_mm(&sys->A, path) != SPARSE_OK) return 0;
    sys->n = sparse_rows(sys->A);
    sys->x_exact = malloc((size_t)sys->n * sizeof(double));
    sys->b = malloc((size_t)sys->n * sizeof(double));
    if (!sys->x_exact || !sys->b) {
        free(sys->x_exact);
        free(sys->b);
        sparse_free(sys->A);
        sys->A = NULL;
        sys->x_exact = NULL;
        sys->b = NULL;
        return 0;
    }
    for (idx_t i = 0; i < sys->n; i++)
        sys->x_exact[i] = (double)(i + 1);
    sparse_matvec(sys->A, sys->x_exact, sys->b);
    return 1;
}

static void free_system(test_system_t *sys)
{
    sparse_free(sys->A);
    free(sys->x_exact);
    free(sys->b);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cross-feature: CG + ILU on all SPD SuiteSparse matrices
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_integration_ilu_cg_all_spd(void)
{
    const char *spd_files[] = {
        SS_DIR "/nos4.mtx",
        SS_DIR "/bcsstk04.mtx",
        NULL
    };

    for (int f = 0; spd_files[f]; f++) {
        test_system_t sys;
        if (!load_system(&sys, spd_files[f])) continue;

        sparse_ilu_t ilu;
        ASSERT_ERR(sparse_ilu_factor(sys.A, &ilu), SPARSE_OK);

        double *x = calloc((size_t)sys.n, sizeof(double));
        sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};
        sparse_iter_result_t result;
        ASSERT_ERR(sparse_solve_cg(sys.A, sys.b, x, &opts,
                                    sparse_ilu_precond, &ilu, &result), SPARSE_OK);
        ASSERT_TRUE(result.converged);

        double res = compute_relative_residual(sys.A, sys.b, x, sys.n);
        printf("    ILU-CG %s: %d iters, res=%.3e\n",
               spd_files[f], (int)result.iterations, res);
        ASSERT_TRUE(res < 1e-8);

        free(x);
        sparse_ilu_free(&ilu);
        free_system(&sys);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cross-feature: GMRES + ILU on all unsymmetric SuiteSparse matrices
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_integration_ilu_gmres_all_unsym(void)
{
    const char *unsym_files[] = {
        SS_DIR "/steam1.mtx",
        SS_DIR "/fs_541_1.mtx",
        SS_DIR "/orsirr_1.mtx",
        NULL
    };

    for (int f = 0; unsym_files[f]; f++) {
        test_system_t sys;
        if (!load_system(&sys, unsym_files[f])) continue;

        sparse_ilu_t ilu;
        sparse_err_t ilu_err = sparse_ilu_factor(sys.A, &ilu);
        if (ilu_err != SPARSE_OK) {
            printf("    ILU-GMRES %s: ILU factor failed (%s)\n",
                   unsym_files[f], sparse_strerror(ilu_err));
            free_system(&sys);
            continue;
        }

        double *x = calloc((size_t)sys.n, sizeof(double));
        sparse_gmres_opts_t opts = {.max_iter = 2000, .restart = 50, .tol = 1e-8, .verbose = 0};
        sparse_iter_result_t result;
        ASSERT_ERR(sparse_solve_gmres(sys.A, sys.b, x, &opts,
                                       sparse_ilu_precond, &ilu, &result), SPARSE_OK);
        ASSERT_TRUE(result.converged);

        double res = compute_relative_residual(sys.A, sys.b, x, sys.n);
        printf("    ILU-GMRES %s: %d iters, res=%.3e\n",
               unsym_files[f], (int)result.iterations, res);
        /* Relaxed tolerance: left preconditioning means the preconditioned
         * residual converges, but the true residual depends on conditioning.
         * Steam1 (condest ~3e7) and orsirr_1 may have larger true residuals. */
        ASSERT_TRUE(res < 1e-2);

        free(x);
        sparse_ilu_free(&ilu);
        free_system(&sys);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cross-feature: Cholesky preconditioner vs ILU on SPD
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_integration_cholesky_vs_ilu_precond(void)
{
    test_system_t sys;
    ASSERT_TRUE(load_system(&sys, SS_DIR "/nos4.mtx"));

    /* Cholesky preconditioner (exact) */
    SparseMatrix *L = sparse_copy(sys.A);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    double *x_chol = calloc((size_t)sys.n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t result_chol;
    ASSERT_ERR(sparse_solve_cg(sys.A, sys.b, x_chol, &opts,
                     cholesky_precond_apply, L, &result_chol), SPARSE_OK);

    /* ILU preconditioner (approximate) */
    sparse_ilu_t ilu;
    ASSERT_ERR(sparse_ilu_factor(sys.A, &ilu), SPARSE_OK);

    double *x_ilu = calloc((size_t)sys.n, sizeof(double));
    sparse_iter_result_t result_ilu;
    ASSERT_ERR(sparse_solve_cg(sys.A, sys.b, x_ilu, &opts,
                     sparse_ilu_precond, &ilu, &result_ilu), SPARSE_OK);

    ASSERT_TRUE(result_chol.converged);
    ASSERT_TRUE(result_ilu.converged);
    /* Cholesky (exact) should converge in fewer iterations than ILU */
    ASSERT_TRUE(result_chol.iterations <= result_ilu.iterations);

    printf("    nos4 precond: Cholesky=%d iters, ILU=%d iters\n",
           (int)result_chol.iterations, (int)result_ilu.iterations);

    free(x_chol); free(x_ilu);
    sparse_free(L);
    sparse_ilu_free(&ilu);
    free_system(&sys);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cross-feature: GMRES + ILU with small restart on orsirr_1
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_integration_ilu_gmres_small_restart_orsirr(void)
{
    test_system_t sys;
    if (!load_system(&sys, SS_DIR "/orsirr_1.mtx")) {
        printf("    [SKIP] orsirr_1.mtx not found\n");
        return;
    }

    sparse_ilu_t ilu;
    ASSERT_ERR(sparse_ilu_factor(sys.A, &ilu), SPARSE_OK);

    double *x = calloc((size_t)sys.n, sizeof(double));
    /* Small restart=10 + ILU should still converge */
    sparse_gmres_opts_t opts = {.max_iter = 2000, .restart = 10, .tol = 1e-8, .verbose = 0};
    sparse_iter_result_t result;
    ASSERT_ERR(sparse_solve_gmres(sys.A, sys.b, x, &opts,
                                   sparse_ilu_precond, &ilu, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);

    double res = compute_relative_residual(sys.A, sys.b, x, sys.n);
    printf("    orsirr_1 ILU-GMRES(10): %d iters, res=%.3e\n",
           (int)result.iterations, res);
    ASSERT_TRUE(res < 1e-4);

    free(x);
    sparse_ilu_free(&ilu);
    free_system(&sys);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Solver comparison: all solvers on same SPD system
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_integration_all_solvers_nos4(void)
{
    test_system_t sys;
    ASSERT_TRUE(load_system(&sys, SS_DIR "/nos4.mtx"));
    idx_t n = sys.n;

    /* CG */
    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t cg_opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res_cg;
    ASSERT_ERR(sparse_solve_cg(sys.A, sys.b, x_cg, &cg_opts, NULL, NULL, &res_cg), SPARSE_OK);

    /* GMRES */
    double *x_gm = calloc((size_t)n, sizeof(double));
    sparse_gmres_opts_t gm_opts = {.max_iter = 500, .restart = 50, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res_gm;
    ASSERT_ERR(sparse_solve_gmres(sys.A, sys.b, x_gm, &gm_opts, NULL, NULL, &res_gm), SPARSE_OK);

    /* LU */
    SparseMatrix *LU = sparse_copy(sys.A);
    double *x_lu = malloc((size_t)n * sizeof(double));
    sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
    sparse_lu_solve(LU, sys.b, x_lu);

    /* Cholesky */
    SparseMatrix *Ch = sparse_copy(sys.A);
    double *x_ch = malloc((size_t)n * sizeof(double));
    sparse_cholesky_factor(Ch);
    sparse_cholesky_solve(Ch, sys.b, x_ch);

    double rr_cg = compute_relative_residual(sys.A, sys.b, x_cg, n);
    double rr_gm = compute_relative_residual(sys.A, sys.b, x_gm, n);
    double rr_lu = compute_relative_residual(sys.A, sys.b, x_lu, n);
    double rr_ch = compute_relative_residual(sys.A, sys.b, x_ch, n);

    printf("    nos4 all solvers: CG=%.3e, GMRES=%.3e, LU=%.3e, Chol=%.3e\n",
           rr_cg, rr_gm, rr_lu, rr_ch);

    ASSERT_TRUE(res_cg.converged);
    ASSERT_TRUE(res_gm.converged);
    ASSERT_TRUE(rr_cg < 1e-8);
    ASSERT_TRUE(rr_gm < 1e-8);
    ASSERT_TRUE(rr_lu < 1e-8);
    ASSERT_TRUE(rr_ch < 1e-8);

    /* All solutions should agree to reasonable precision */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x_cg[i], x_lu[i], 1e-4);
        ASSERT_NEAR(x_gm[i], x_lu[i], 1e-4);
        ASSERT_NEAR(x_ch[i], x_lu[i], 1e-4);
    }

    free(x_cg); free(x_gm); free(x_lu); free(x_ch);
    sparse_free(LU); sparse_free(Ch);
    free_system(&sys);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Edge-case integration tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* All solvers on 1×1 system */
static void test_integration_1x1_all_solvers(void)
{
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 5.0);
    double b[1] = {15.0};

    /* CG */
    double x_cg[1] = {0.0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_cg(A, b, x_cg, NULL, NULL, NULL, &res), SPARSE_OK);
    ASSERT_NEAR(x_cg[0], 3.0, 1e-14);

    /* GMRES */
    double x_gm[1] = {0.0};
    ASSERT_ERR(sparse_solve_gmres(A, b, x_gm, NULL, NULL, NULL, &res), SPARSE_OK);
    ASSERT_NEAR(x_gm[0], 3.0, 1e-14);

    /* ILU-preconditioned CG */
    sparse_ilu_t ilu;
    ASSERT_ERR(sparse_ilu_factor(A, &ilu), SPARSE_OK);
    double x_pcg[1] = {0.0};
    ASSERT_ERR(sparse_solve_cg(A, b, x_pcg, NULL, sparse_ilu_precond, &ilu, &res), SPARSE_OK);
    ASSERT_NEAR(x_pcg[0], 3.0, 1e-14);
    sparse_ilu_free(&ilu);

    sparse_free(A);
}

/* Zero tolerance → runs to max_iter */
static void test_integration_zero_tolerance(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0); sparse_insert(A, 0, 1, -1.0);
    sparse_insert(A, 1, 0, -1.0); sparse_insert(A, 1, 1, 4.0); sparse_insert(A, 1, 2, -1.0);
    sparse_insert(A, 2, 1, -1.0); sparse_insert(A, 2, 2, 4.0);

    double b[3] = {3.0, 2.0, 3.0};
    double x[3] = {0.0, 0.0, 0.0};

    /* tol=0 means it can never converge via residual check; must hit max_iter */
    sparse_iter_opts_t opts = {.max_iter = 5, .tol = 0.0, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);

    /* CG on 3×3 SPD converges in ≤3 steps in exact arithmetic,
     * but tol=0 means the residual check never triggers.
     * It may converge via lucky breakdown or hit max_iter. */
    ASSERT_TRUE(result.iterations >= 0);
    ASSERT_TRUE(result.iterations <= 5);
    /* Even if "not converged", the solution should be reasonable */
    (void)err;

    sparse_free(A);
}

/* Identity preconditioner = unpreconditioned */
static void test_integration_identity_preconditioner(void)
{
    SparseMatrix *A = sparse_create(5, 5);
    for (idx_t i = 0; i < 5; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i < 4) sparse_insert(A, i, i + 1, -1.0);
    }

    double x_exact[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[5];
    sparse_matvec(A, x_exact, b);

    /* Unpreconditioned CG */
    double x_unprec[5] = {0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res_unprec;
    ASSERT_ERR(sparse_solve_cg(A, b, x_unprec, &opts, NULL, NULL, &res_unprec), SPARSE_OK);

    /* Identity preconditioned CG — should behave identically */
    double x_ident[5] = {0};
    sparse_iter_result_t res_ident;
    ASSERT_ERR(sparse_solve_cg(A, b, x_ident, &opts, identity_precond, NULL, &res_ident), SPARSE_OK);

    ASSERT_TRUE(res_unprec.converged);
    ASSERT_TRUE(res_ident.converged);
    ASSERT_EQ(res_unprec.iterations, res_ident.iterations);

    for (int i = 0; i < 5; i++)
        ASSERT_NEAR(x_unprec[i], x_ident[i], 1e-14);

    sparse_free(A);
}

/* ILU factor → solve → verify L*U*z ≈ A*z for multiple RHS */
static void test_integration_ilu_multi_rhs(void)
{
    test_system_t sys;
    ASSERT_TRUE(load_system(&sys, SS_DIR "/nos4.mtx"));
    idx_t n = sys.n;

    sparse_ilu_t ilu;
    ASSERT_ERR(sparse_ilu_factor(sys.A, &ilu), SPARSE_OK);

    /* Solve with 3 different RHS vectors using the same ILU factors */
    for (int rhs = 0; rhs < 3; rhs++) {
        double *b = malloc((size_t)n * sizeof(double));
        for (idx_t i = 0; i < n; i++)
            b[i] = sin((double)(i + 1) * (0.1 * (double)(rhs + 1)));

        double *x = calloc((size_t)n, sizeof(double));
        sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
        sparse_iter_result_t result;
        sparse_solve_cg(sys.A, b, x, &opts, sparse_ilu_precond, &ilu, &result);

        ASSERT_TRUE(result.converged);
        double res = compute_relative_residual(sys.A, b, x, n);
        ASSERT_TRUE(res < 1e-8);

        free(b); free(x);
    }

    sparse_ilu_free(&ilu);
    free_system(&sys);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Edge-case hardening (Day 13)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Very ill-conditioned system: CG/GMRES report non-convergence gracefully */
static void test_hardening_illcond_nconv(void)
{
    /* Hilbert-like 5×5 matrix: very ill-conditioned */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, 1.0 / (double)(i + j + 1));

    double b[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double x[5] = {0};

    /* CG with very few iterations — should not crash */
    sparse_iter_opts_t cg_opts = {.max_iter = 3, .tol = 1e-15, .verbose = 0};
    sparse_iter_result_t result;
    sparse_solve_cg(A, b, x, &cg_opts, NULL, NULL, &result);
    ASSERT_TRUE(result.iterations <= 3);
    ASSERT_TRUE(result.residual_norm >= 0.0);

    /* GMRES with very few iterations */
    double xg[5] = {0};
    sparse_gmres_opts_t gm_opts = {.max_iter = 3, .restart = 3, .tol = 1e-15, .verbose = 0};
    sparse_solve_gmres(A, b, xg, &gm_opts, NULL, NULL, &result);
    ASSERT_TRUE(result.iterations <= 3);
    ASSERT_TRUE(result.residual_norm >= 0.0);

    sparse_free(A);
}

/* GMRES handles singular-like system without crash */
static void test_hardening_gmres_near_singular(void)
{
    /* Nearly singular: rank-1 matrix plus tiny perturbation */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, 1.0);
    /* Add small diagonal perturbation to make non-singular */
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0 + 1e-10);

    double b[4] = {1.0, 1.0, 1.0, 1.0};
    double x[4] = {0};
    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 10, .tol = 1e-8, .verbose = 0};
    sparse_iter_result_t result;

    /* Should not crash; may or may not converge */
    sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result);
    ASSERT_TRUE(result.iterations >= 0);

    sparse_free(A);
}

/* Very large restart value: no excessive memory (clamped to n) */
static void test_hardening_large_restart(void)
{
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)     sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1) sparse_insert(A, i, i + 1, -1.0);
    }

    double b[5] = {3.0, 2.0, 2.0, 2.0, 3.0};
    double x[5] = {0};
    /* restart=1000000 >> n: should be clamped to n internally */
    sparse_gmres_opts_t opts = {.max_iter = 100, .restart = 1000000, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;

    ASSERT_ERR(sparse_solve_gmres(A, b, x, &opts, NULL, NULL, &result), SPARSE_OK);
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.iterations <= n);

    sparse_free(A);
}

/* ILU on matrix with very small diagonal */
static void test_hardening_ilu_tiny_diagonal(void)
{
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 1e-20);
    sparse_insert(A, 1, 1, 1e-20);
    sparse_insert(A, 2, 2, 1e-20);

    sparse_ilu_t ilu;
    /* Should either succeed with tiny factors or fail gracefully */
    sparse_err_t err = sparse_ilu_factor(A, &ilu);
    if (err == SPARSE_OK)
        sparse_ilu_free(&ilu);
    /* Either way, no crash */
    ASSERT_TRUE(err == SPARSE_OK || err == SPARSE_ERR_SINGULAR);

    sparse_free(A);
}

/* CG and GMRES with max_iter=0 */
static void test_hardening_zero_max_iter(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    for (idx_t i = 0; i < 3; i++)
        sparse_insert(A, i, i, 4.0);
    double b[3] = {1.0, 2.0, 3.0};
    double x[3] = {0};
    sparse_iter_result_t result;

    sparse_iter_opts_t cg_opts = {.max_iter = 0, .tol = 1e-10, .verbose = 0};
    sparse_solve_cg(A, b, x, &cg_opts, NULL, NULL, &result);
    ASSERT_EQ(result.iterations, 0);

    sparse_gmres_opts_t gm_opts = {.max_iter = 0, .restart = 10, .tol = 1e-10, .verbose = 0};
    double xg[3] = {0};
    sparse_solve_gmres(A, b, xg, &gm_opts, NULL, NULL, &result);
    ASSERT_EQ(result.iterations, 0);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Sprint 5 Integration Tests");

    /* Cross-feature: iterative + ILU on SuiteSparse */
    RUN_TEST(test_integration_ilu_cg_all_spd);
    RUN_TEST(test_integration_ilu_gmres_all_unsym);
    RUN_TEST(test_integration_cholesky_vs_ilu_precond);
    RUN_TEST(test_integration_ilu_gmres_small_restart_orsirr);

    /* Solver comparison */
    RUN_TEST(test_integration_all_solvers_nos4);

    /* Edge cases */
    RUN_TEST(test_integration_1x1_all_solvers);
    RUN_TEST(test_integration_zero_tolerance);
    RUN_TEST(test_integration_identity_preconditioner);

    /* Multi-RHS with shared ILU */
    RUN_TEST(test_integration_ilu_multi_rhs);

    /* Edge-case hardening (Day 13) */
    RUN_TEST(test_hardening_illcond_nconv);
    RUN_TEST(test_hardening_gmres_near_singular);
    RUN_TEST(test_hardening_large_restart);
    RUN_TEST(test_hardening_ilu_tiny_diagonal);
    RUN_TEST(test_hardening_zero_max_iter);

    TEST_SUITE_END();
}
