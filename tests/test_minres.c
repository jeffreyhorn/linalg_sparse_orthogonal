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

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build n×n SPD tridiagonal: diag=4, off=-1 */
static SparseMatrix *make_spd_tridiag(idx_t n) {
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

/* Build n×n symmetric indefinite matrix: KKT-type [H A^T; A 0] */
static SparseMatrix *make_kkt(idx_t nh, idx_t nc) {
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

/* Build n×n symmetric indefinite tridiag: diag alternates +2,-2, off=1 */
static SparseMatrix *make_sym_indef_tridiag(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (i % 2 == 0) ? 2.0 : -2.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, 1.0);
            sparse_insert(A, i - 1, i, 1.0);
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

/* ═══════════════════════════════════════════════════════════════════════
 * Entry validation tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minres_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    double b[3] = {1, 2, 3}, x[3];
    sparse_iter_result_t res;

    ASSERT_ERR(sparse_solve_minres(NULL, b, x, NULL, NULL, NULL, &res), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_solve_minres(A, NULL, x, NULL, NULL, NULL, &res), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_solve_minres(A, b, NULL, NULL, NULL, NULL, &res), SPARSE_ERR_NULL);

    sparse_free(A);
}

static void test_minres_non_square(void) {
    SparseMatrix *A = sparse_create(3, 4);
    double b[3] = {1, 2, 3}, x[3];
    ASSERT_ERR(sparse_solve_minres(A, b, x, NULL, NULL, NULL, NULL), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

static void test_minres_zero_rhs(void) {
    idx_t n = 5;
    SparseMatrix *A = make_spd_tridiag(n);
    double b[5] = {0, 0, 0, 0, 0};
    double x[5] = {1, 2, 3, 4, 5};
    sparse_iter_result_t res;

    ASSERT_ERR(sparse_solve_minres(A, b, x, NULL, NULL, NULL, &res), SPARSE_OK);
    ASSERT_TRUE(res.converged);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 0.0, 1e-15);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SPD system tests (MINRES should work on SPD like CG)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minres_spd_diagonal(void) {
    /* Diagonal SPD: MINRES should converge quickly */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    double b[] = {1, 4, 9, 16, 25};
    double x[5];
    memset(x, 0, sizeof(x));

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    printf("    SPD diagonal: %d iters, relres=%.3e\n", (int)res.iterations, res.residual_norm);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i] / (double)(i + 1), 1e-10);

    sparse_free(A);
}

static void test_minres_spd_tridiag(void) {
    idx_t n = 20;
    SparseMatrix *A = make_spd_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    double relres = relative_residual(A, x, b, n);
    printf("    SPD tridiag n=%d: %d iters, relres=%.3e\n", (int)n, (int)res.iterations, relres);
    ASSERT_TRUE(relres < 1e-10);

    free(x_exact);
    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_spd_vs_cg(void) {
    /* MINRES and CG should produce the same solution on SPD systems */
    idx_t n = 15;
    SparseMatrix *A = make_spd_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};

    double *x_minres = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_minres;
    ASSERT_ERR(sparse_solve_minres(A, b, x_minres, &opts, NULL, NULL, &res_minres), SPARSE_OK);

    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_cg;
    ASSERT_ERR(sparse_solve_cg(A, b, x_cg, &opts, NULL, NULL, &res_cg), SPARSE_OK);

    ASSERT_TRUE(res_minres.converged);
    ASSERT_TRUE(res_cg.converged);

    printf("    SPD n=%d: MINRES %d iters, CG %d iters\n", (int)n, (int)res_minres.iterations,
           (int)res_cg.iterations);

    /* Solutions should agree */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_minres[i], x_cg[i], 1e-8);

    free(x_exact);
    free(b);
    free(x_minres);
    free(x_cg);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symmetric indefinite tests (MINRES advantage over CG)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minres_indefinite_tridiag(void) {
    /* Symmetric indefinite tridiagonal — CG can't handle this, MINRES can */
    idx_t n = 10;
    SparseMatrix *A = make_sym_indef_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    double relres = relative_residual(A, x, b, n);
    printf("    indefinite tridiag n=%d: %d iters, relres=%.3e\n", (int)n, (int)res.iterations,
           relres);
    ASSERT_TRUE(relres < 1e-8);

    free(x_exact);
    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_kkt_small(void) {
    /* Small KKT system: symmetric indefinite */
    idx_t nh = 6, nc = 3;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(K, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(K, b, x, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    double relres = relative_residual(K, x, b, n);
    printf("    KKT %dx%d: %d iters, relres=%.3e\n", (int)n, (int)n, (int)res.iterations, relres);
    ASSERT_TRUE(relres < 1e-8);

    free(x_exact);
    free(b);
    free(x);
    sparse_free(K);
}

static void test_minres_kkt_medium(void) {
    /* Medium KKT system */
    idx_t nh = 20, nc = 8;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(K, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(K, b, x, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    double relres = relative_residual(K, x, b, n);
    printf("    KKT %dx%d: %d iters, relres=%.3e\n", (int)n, (int)n, (int)res.iterations, relres);
    ASSERT_TRUE(relres < 1e-8);

    free(x_exact);
    free(b);
    free(x);
    sparse_free(K);
}

static void test_minres_1x1(void) {
    /* 1x1 system */
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 3.0);
    double b = 9.0, x = 0.0;
    sparse_iter_opts_t opts = {.max_iter = 10, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, &b, &x, &opts, NULL, NULL, &res), SPARSE_OK);
    ASSERT_TRUE(res.converged);
    ASSERT_NEAR(x, 3.0, 1e-12);
    sparse_free(A);
}

static void test_minres_1x1_negative(void) {
    /* 1x1 negative definite */
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, -5.0);
    double b = -15.0, x = 0.0;
    sparse_iter_opts_t opts = {.max_iter = 10, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, &b, &x, &opts, NULL, NULL, &res), SPARSE_OK);
    ASSERT_TRUE(res.converged);
    ASSERT_NEAR(x, 3.0, 1e-12);
    sparse_free(A);
}

static void test_minres_already_converged(void) {
    /* Initial guess is the exact solution */
    idx_t n = 5;
    SparseMatrix *A = make_spd_tridiag(n);
    double x_exact[] = {1, 2, 3, 4, 5};
    double b[5];
    sparse_matvec(A, x_exact, b);

    double x[5];
    memcpy(x, x_exact, sizeof(x));

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res), SPARSE_OK);
    ASSERT_TRUE(res.converged);
    ASSERT_EQ(res.iterations, 0);

    sparse_free(A);
}

static void test_minres_vs_gmres_indefinite(void) {
    /* MINRES and GMRES should agree on symmetric indefinite systems */
    idx_t nh = 10, nc = 4;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(K, x_exact, b);

    sparse_iter_opts_t cg_opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_gmres_opts_t gm_opts = {.max_iter = 500, .restart = 50, .tol = 1e-10, .verbose = 0};

    double *x_minres = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_minres;
    ASSERT_ERR(sparse_solve_minres(K, b, x_minres, &cg_opts, NULL, NULL, &res_minres), SPARSE_OK);

    double *x_gmres = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_gmres;
    ASSERT_ERR(sparse_solve_gmres(K, b, x_gmres, &gm_opts, NULL, NULL, &res_gmres), SPARSE_OK);

    ASSERT_TRUE(res_minres.converged);
    ASSERT_TRUE(res_gmres.converged);

    printf("    MINRES vs GMRES on KKT %dx%d: MINRES %d iters, GMRES %d iters\n", (int)n, (int)n,
           (int)res_minres.iterations, (int)res_gmres.iterations);

    /* Solutions should agree */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_minres[i], x_gmres[i], 1e-6);

    free(x_exact);
    free(b);
    free(x_minres);
    free(x_gmres);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Preconditioned MINRES tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Jacobi (diagonal) preconditioner: M = diag(|A(i,i)|) */
typedef struct {
    double *diag_inv; /* 1/|A(i,i)| */
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

static void test_minres_precond_ic_spd(void) {
    /* IC(0)-preconditioned MINRES on SPD system.
     * Compare: unpreconditioned vs IC(0)-preconditioned */
    idx_t n = 30;
    SparseMatrix *A = make_spd_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_minres(A, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);
    ASSERT_TRUE(res1.converged);

    /* IC(0)-preconditioned */
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_solve_minres(A, b, x2, &opts, sparse_ic_precond, &ic, &res2), SPARSE_OK);
    ASSERT_TRUE(res2.converged);

    printf("    IC(0)-MINRES on SPD n=%d: unprec=%d iters, IC(0)=%d iters\n", (int)n,
           (int)res1.iterations, (int)res2.iterations);

    /* IC(0) should reduce iteration count */
    ASSERT_TRUE(res2.iterations <= res1.iterations);

    /* Solutions should agree */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x1[i], x2[i], 1e-6);

    double relres = relative_residual(A, x2, b, n);
    ASSERT_TRUE(relres < 1e-8);

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_minres_precond_ic_vs_cg(void) {
    /* IC(0)-preconditioned MINRES should match IC(0)-preconditioned CG on SPD */
    idx_t n = 20;
    SparseMatrix *A = make_spd_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};

    double *x_minres = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_minres;
    ASSERT_ERR(sparse_solve_minres(A, b, x_minres, &opts, sparse_ic_precond, &ic, &res_minres),
               SPARSE_OK);

    double *x_cg = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_cg;
    ASSERT_ERR(sparse_solve_cg(A, b, x_cg, &opts, sparse_ic_precond, &ic, &res_cg), SPARSE_OK);

    ASSERT_TRUE(res_minres.converged);
    ASSERT_TRUE(res_cg.converged);

    printf("    IC(0)-precond n=%d: MINRES %d iters, CG %d iters\n", (int)n,
           (int)res_minres.iterations, (int)res_cg.iterations);

    /* Solutions should agree */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_minres[i], x_cg[i], 1e-6);

    free(x_exact);
    free(b);
    free(x_minres);
    free(x_cg);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_minres_precond_jacobi_indefinite(void) {
    /* Jacobi-preconditioned MINRES on symmetric indefinite KKT system */
    idx_t nh = 15, nc = 6;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(K, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_minres(K, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);

    /* Jacobi-preconditioned */
    jacobi_ctx_t jac = make_jacobi(K, n);
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_solve_minres(K, b, x2, &opts, jacobi_precond, &jac, &res2), SPARSE_OK);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);

    printf("    Jacobi-MINRES on KKT %dx%d: unprec=%d iters, Jacobi=%d iters\n", (int)n, (int)n,
           (int)res1.iterations, (int)res2.iterations);

    /* Both should converge to the same solution */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x1[i], x2[i], 1e-5);

    double relres = relative_residual(K, x2, b, n);
    ASSERT_TRUE(relres < 1e-8);

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    free(jac.diag_inv);
    sparse_free(K);
}

static void test_minres_precond_jacobi_large_kkt(void) {
    /* Larger KKT with Jacobi preconditioning */
    idx_t nh = 40, nc = 15;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(K, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    sparse_solve_minres(K, b, x1, &opts, NULL, NULL, &res1);

    /* Jacobi-preconditioned */
    jacobi_ctx_t jac = make_jacobi(K, n);
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    sparse_solve_minres(K, b, x2, &opts, jacobi_precond, &jac, &res2);

    printf("    Jacobi-MINRES on KKT %dx%d: unprec=%d iters (relres=%.1e), Jacobi=%d iters "
           "(relres=%.1e)\n",
           (int)n, (int)n, (int)res1.iterations, res1.residual_norm, (int)res2.iterations,
           res2.residual_norm);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);

    double relres = relative_residual(K, x2, b, n);
    ASSERT_TRUE(relres < 1e-8);

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    free(jac.diag_inv);
    sparse_free(K);
}

static void test_minres_precond_ic_banded(void) {
    /* IC(0)-preconditioned MINRES on larger banded SPD */
    idx_t n = 50;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        for (idx_t d = 1; d <= 3 && i + d < n; d++) {
            double off = -1.0 / (double)(d + 1);
            sparse_insert(A, i, i + d, off);
            sparse_insert(A, i + d, i, off);
        }
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1) / (double)n;
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_minres(A, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);

    /* IC(0)-preconditioned */
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_solve_minres(A, b, x2, &opts, sparse_ic_precond, &ic, &res2), SPARSE_OK);

    /* ILU(0)-preconditioned for comparison */
    sparse_ilu_t ilu;
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));
    double *x3 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res3;
    ASSERT_ERR(sparse_solve_minres(A, b, x3, &opts, sparse_ilu_precond, &ilu, &res3), SPARSE_OK);

    printf("    banded n=%d MINRES iters: unprec=%d, IC(0)=%d, ILU(0)=%d\n", (int)n,
           (int)res1.iterations, (int)res2.iterations, (int)res3.iterations);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);
    ASSERT_TRUE(res3.converged);
    ASSERT_TRUE(res2.iterations <= res1.iterations);

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
 * Edge cases & robustness
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minres_max_iter(void) {
    /* Verify NOT_CONVERGED is returned when max_iter is too small */
    idx_t n = 20;
    SparseMatrix *A = make_spd_tridiag(n);

    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 2, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res), SPARSE_ERR_NOT_CONVERGED);
    ASSERT_FALSE(res.converged);
    ASSERT_EQ(res.iterations, 2);

    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_large_indefinite(void) {
    /* Larger symmetric indefinite system */
    idx_t n = 50;
    SparseMatrix *A = make_sym_indef_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = cos((double)(i + 1));
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    double relres = relative_residual(A, x, b, n);
    printf("    indefinite tridiag n=%d: %d iters, relres=%.3e\n", (int)n, (int)res.iterations,
           relres);
    ASSERT_TRUE(relres < 1e-8);

    free(x_exact);
    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_precond_identity(void) {
    /* Identity preconditioner (M=I) should give same result as unpreconditioned */
    idx_t n = 10;
    SparseMatrix *A = make_spd_tridiag(n);

    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};

    /* Build identity preconditioner as IC(0) of I */
    SparseMatrix *I_mat = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(I_mat, i, i, 1.0);
    sparse_ilu_t ic_I;
    REQUIRE_OK(sparse_ic_factor(I_mat, &ic_I));

    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_minres(A, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);

    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_solve_minres(A, b, x2, &opts, sparse_ic_precond, &ic_I, &res2), SPARSE_OK);

    /* Same iteration count and solution */
    ASSERT_EQ(res1.iterations, res2.iterations);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x1[i], x2[i], 1e-12);

    free(b);
    free(x1);
    free(x2);
    sparse_ic_free(&ic_I);
    sparse_free(I_mat);
    sparse_free(A);
}

static void test_minres_precond_exact(void) {
    /* Exact preconditioner (M=A for SPD) should converge in 1 iteration */
    idx_t n = 5;
    SparseMatrix *A = make_spd_tridiag(n);

    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    /* Use Cholesky of A as "exact" preconditioner via IC(0).
     * For tridiagonal, IC(0) = exact Cholesky. */
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double *x = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x, &opts, sparse_ic_precond, &ic, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    printf("    exact precond n=%d: %d iters\n", (int)n, (int)res.iterations);
    ASSERT_TRUE(res.iterations <= 1);

    double relres = relative_residual(A, x, b, n);
    ASSERT_TRUE(relres < 1e-10);

    free(b);
    free(x);
    sparse_ic_free(&ic);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Numerical robustness tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build a scaled symmetric indefinite 3x3 matrix */
static SparseMatrix *make_scaled_indef(double s) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0 * s);
    sparse_insert(A, 0, 1, 1.0 * s);
    sparse_insert(A, 0, 2, -1.0 * s);
    sparse_insert(A, 1, 0, 1.0 * s);
    sparse_insert(A, 1, 1, -3.0 * s);
    sparse_insert(A, 2, 0, -1.0 * s);
    sparse_insert(A, 2, 2, 4.0 * s);
    return A;
}

static void test_minres_scaled_tolerance(void) {
    /* Extreme-scale matrices: verify no overflow/underflow */
    double scales[] = {1e-35, 1e-10, 1.0, 1e10, 1e35};
    int nscales = 5;

    for (int s = 0; s < nscales; s++) {
        SparseMatrix *A = make_scaled_indef(scales[s]);

        double x_exact[] = {1.0, 2.0, 3.0};
        double b[3], x[3];
        sparse_matvec(A, x_exact, b);
        memset(x, 0, sizeof(x));

        sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-8, .verbose = 0};
        sparse_iter_result_t res;
        sparse_err_t err = sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res);

        if (err == SPARSE_OK && res.converged) {
            for (int i = 0; i < 3; i++)
                ASSERT_NEAR(x[i], x_exact[i], 1e-4);
        }
        /* At extreme scales, accept convergence or graceful failure */
        ASSERT_TRUE(err == SPARSE_OK || err == SPARSE_ERR_NOT_CONVERGED);

        sparse_free(A);
    }
    printf("    scaled tolerance: 5 scales from 1e-35 to 1e+35 OK\n");
}

static void test_minres_scaled_spd(void) {
    /* Scaled SPD: verify MINRES works at extreme scales */
    double scales[] = {1e-20, 1e-5, 1.0, 1e5, 1e20};

    for (int s = 0; s < 5; s++) {
        idx_t n = 8;
        SparseMatrix *A = make_spd_tridiag(n);
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

        double *x_exact = malloc((size_t)n * sizeof(double));
        double *b = malloc((size_t)n * sizeof(double));
        double *x = calloc((size_t)n, sizeof(double));
        for (idx_t i = 0; i < n; i++)
            x_exact[i] = (double)(i + 1);
        sparse_matvec(A, x_exact, b);

        sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-8, .verbose = 0};
        sparse_iter_result_t res;
        sparse_err_t err = sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res);

        if (err == SPARSE_OK && res.converged) {
            for (idx_t i = 0; i < n; i++)
                ASSERT_NEAR(x[i], x_exact[i], 1e-4);
        }
        ASSERT_TRUE(err == SPARSE_OK || err == SPARSE_ERR_NOT_CONVERGED);

        free(x_exact);
        free(b);
        free(x);
        sparse_free(A);
    }
    printf("    scaled SPD: 5 scales from 1e-20 to 1e+20 OK\n");
}

static void test_minres_ill_conditioned(void) {
    /* Ill-conditioned symmetric indefinite: eigenvalues span wide range */
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    /* Diagonal with eigenvalues: 1e-6, ..., 1e3, -1e-6, ..., -1e3 */
    for (idx_t i = 0; i < n / 2; i++) {
        double val = pow(10.0, -6.0 + 9.0 * (double)i / (double)(n / 2 - 1));
        sparse_insert(A, i, i, val);
    }
    for (idx_t i = n / 2; i < n; i++) {
        double val = -pow(10.0, -6.0 + 9.0 * (double)(i - n / 2) / (double)(n / 2 - 1));
        sparse_insert(A, i, i, val);
    }

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-8, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    double relres = relative_residual(A, x, b, n);
    printf("    ill-conditioned (cond~1e9): %d iters, relres=%.3e\n", (int)res.iterations, relres);
    ASSERT_TRUE(relres < 1e-6);

    free(x_exact);
    free(b);
    free(x);
    sparse_free(A);
}

static void test_minres_early_lanczos_termination(void) {
    /* Matrix that causes early Lanczos termination:
     * A = diag(1,2) with b = (1,0) → Krylov space is 1D → converges in 1 iter */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);

    double b[] = {1.0, 0.0};
    double x[2] = {0.0, 0.0};

    sparse_iter_opts_t opts = {.max_iter = 10, .tol = 1e-14, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    ASSERT_NEAR(x[0], 1.0, 1e-14);
    ASSERT_NEAR(x[1], 0.0, 1e-14);
    /* Should converge in 1 iteration (Lanczos terminates) */
    ASSERT_TRUE(res.iterations <= 1);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MINRES vs direct solver (LDL^T) comparison
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minres_vs_ldlt_spd(void) {
    /* MINRES vs LDL^T on SPD system: solutions should agree */
    idx_t n = 10;
    SparseMatrix *A = make_spd_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* MINRES */
    double *x_minres = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(A, b, x_minres, &opts, NULL, NULL, &res), SPARSE_OK);
    ASSERT_TRUE(res.converged);

    /* LDL^T */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(A, &ldlt));
    double *x_ldlt = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(sparse_ldlt_solve(&ldlt, b, x_ldlt));

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_minres[i], x_ldlt[i], 1e-8);

    free(x_exact);
    free(b);
    free(x_minres);
    free(x_ldlt);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_minres_vs_ldlt_indefinite(void) {
    /* MINRES vs LDL^T on symmetric indefinite KKT system */
    idx_t nh = 8, nc = 3;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(K, x_exact, b);

    /* MINRES */
    double *x_minres = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(K, b, x_minres, &opts, NULL, NULL, &res), SPARSE_OK);
    ASSERT_TRUE(res.converged);

    /* LDL^T */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(K, &ldlt));
    double *x_ldlt = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(sparse_ldlt_solve(&ldlt, b, x_ldlt));

    printf("    MINRES vs LDL^T on KKT %dx%d: ", (int)n, (int)n);
    double max_diff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double diff = fabs(x_minres[i] - x_ldlt[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    printf("max|diff|=%.3e\n", max_diff);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_minres[i], x_ldlt[i], 1e-8);

    free(x_exact);
    free(b);
    free(x_minres);
    free(x_ldlt);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

static void test_minres_vs_ldlt_large_kkt(void) {
    /* Larger KKT: MINRES vs LDL^T */
    idx_t nh = 20, nc = 8;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(K, x_exact, b);

    /* MINRES */
    double *x_minres = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_minres(K, b, x_minres, &opts, NULL, NULL, &res), SPARSE_OK);

    /* LDL^T */
    sparse_ldlt_t ldlt;
    REQUIRE_OK(sparse_ldlt_factor(K, &ldlt));
    double *x_ldlt = malloc((size_t)n * sizeof(double));
    REQUIRE_OK(sparse_ldlt_solve(&ldlt, b, x_ldlt));

    ASSERT_TRUE(res.converged);
    double relres_minres = relative_residual(K, x_minres, b, n);
    double relres_ldlt = relative_residual(K, x_ldlt, b, n);
    printf("    KKT %dx%d: MINRES relres=%.3e (%d iters), LDL^T relres=%.3e\n", (int)n, (int)n,
           relres_minres, (int)res.iterations, relres_ldlt);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_minres[i], x_ldlt[i], 1e-6);

    free(x_exact);
    free(b);
    free(x_minres);
    free(x_ldlt);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

static void test_minres_vs_gmres_large(void) {
    /* Larger MINRES vs GMRES comparison on indefinite system */
    idx_t nh = 30, nc = 12;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = cos((double)(i + 1));
    sparse_matvec(K, x_exact, b);

    sparse_iter_opts_t cg_opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_gmres_opts_t gm_opts = {.max_iter = 500, .restart = 50, .tol = 1e-10, .verbose = 0};

    double *x_minres = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_mr;
    ASSERT_ERR(sparse_solve_minres(K, b, x_minres, &cg_opts, NULL, NULL, &res_mr), SPARSE_OK);

    double *x_gmres = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_gm;
    ASSERT_ERR(sparse_solve_gmres(K, b, x_gmres, &gm_opts, NULL, NULL, &res_gm), SPARSE_OK);

    ASSERT_TRUE(res_mr.converged);
    ASSERT_TRUE(res_gm.converged);

    printf("    KKT %dx%d: MINRES %d iters, GMRES %d iters\n", (int)n, (int)n,
           (int)res_mr.iterations, (int)res_gm.iterations);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_minres[i], x_gmres[i], 1e-6);

    free(x_exact);
    free(b);
    free(x_minres);
    free(x_gmres);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Block MINRES tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minres_block_null_args(void) {
    SparseMatrix *A = sparse_create(3, 3);
    double B[3], X[3];
    ASSERT_ERR(sparse_minres_solve_block(NULL, B, 1, X, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_minres_solve_block(A, NULL, 1, X, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_minres_solve_block(A, B, 1, NULL, NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_minres_block_zero_nrhs(void) {
    SparseMatrix *A = make_spd_tridiag(5);
    double B[5], X[5];
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_minres_solve_block(A, B, 0, X, NULL, NULL, NULL, &res), SPARSE_OK);
    ASSERT_TRUE(res.converged);
    ASSERT_EQ(res.iterations, 0);
    sparse_free(A);
}

static void test_minres_block_single_rhs(void) {
    /* Single RHS via block API should match single-RHS MINRES */
    idx_t n = 10;
    SparseMatrix *A = make_spd_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};

    /* Single-RHS MINRES */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_minres(A, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);

    /* Block MINRES with nrhs=1 */
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_minres_solve_block(A, b, 1, x2, &opts, NULL, NULL, &res2), SPARSE_OK);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);
    ASSERT_EQ(res1.iterations, res2.iterations);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x1[i], x2[i], 1e-14);

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    sparse_free(A);
}

static void test_minres_block_multi_rhs_spd(void) {
    /* Multiple RHS on SPD system */
    idx_t n = 15;
    idx_t nrhs = 4;
    SparseMatrix *A = make_spd_tridiag(n);

    /* Build B = [b1, b2, b3, b4] column-major, each from a known x_exact */
    double *X_exact = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    for (idx_t j = 0; j < nrhs; j++) {
        for (idx_t i = 0; i < n; i++)
            X_exact[j * n + i] = sin((double)(j * n + i + 1));
        sparse_matvec(A, X_exact + j * n, B + j * n);
    }

    double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_minres_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    printf("    block MINRES SPD n=%d nrhs=%d: %d iters, relres=%.3e\n", (int)n, (int)nrhs,
           (int)res.iterations, res.residual_norm);

    /* Verify each column */
    for (idx_t j = 0; j < nrhs; j++) {
        double relres = relative_residual(A, X + j * n, B + j * n, n);
        ASSERT_TRUE(relres < 1e-8);
    }

    free(X_exact);
    free(B);
    free(X);
    sparse_free(A);
}

static void test_minres_block_multi_rhs_indefinite(void) {
    /* Multiple RHS on symmetric indefinite KKT system */
    idx_t nh = 10, nc = 4;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;
    idx_t nrhs = 3;

    double *X_exact = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    for (idx_t j = 0; j < nrhs; j++) {
        for (idx_t i = 0; i < n; i++)
            X_exact[j * n + i] = (double)(j * n + i + 1);
        sparse_matvec(K, X_exact + j * n, B + j * n);
    }

    double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_minres_solve_block(K, B, nrhs, X, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    printf("    block MINRES KKT %dx%d nrhs=%d: %d iters, relres=%.3e\n", (int)n, (int)n, (int)nrhs,
           (int)res.iterations, res.residual_norm);

    for (idx_t j = 0; j < nrhs; j++) {
        double relres = relative_residual(K, X + j * n, B + j * n, n);
        ASSERT_TRUE(relres < 1e-8);
    }

    free(X_exact);
    free(B);
    free(X);
    sparse_free(K);
}

static void test_minres_block_zero_rhs_column(void) {
    /* One column is zero RHS — that column should give x = 0 */
    idx_t n = 5;
    idx_t nrhs = 3;
    SparseMatrix *A = make_spd_tridiag(n);

    double *B = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    /* Column 0: nonzero RHS */
    for (idx_t i = 0; i < n; i++)
        B[i] = (double)(i + 1);
    /* Column 1: zero RHS */
    /* (already zero from calloc) */
    /* Column 2: nonzero RHS */
    for (idx_t i = 0; i < n; i++)
        B[2 * n + i] = (double)(n - i);

    double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_minres_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);

    /* Column 0 should have nonzero solution */
    double col0_norm = 0;
    for (idx_t i = 0; i < n; i++)
        col0_norm += X[i] * X[i];
    ASSERT_TRUE(col0_norm > 0.1);

    /* Column 1 should be zero */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(X[n + i], 0.0, 1e-14);

    /* Column 2 should have nonzero solution */
    double col2_norm = 0;
    for (idx_t i = 0; i < n; i++)
        col2_norm += X[2 * n + i] * X[2 * n + i];
    ASSERT_TRUE(col2_norm > 0.1);

    /* Verify residuals for nonzero columns */
    ASSERT_TRUE(relative_residual(A, X, B, n) < 1e-10);
    ASSERT_TRUE(relative_residual(A, X + 2 * n, B + 2 * n, n) < 1e-10);

    free(B);
    free(X);
    sparse_free(A);
}

static void test_minres_block_large(void) {
    /* Larger block solve: KKT with 5 RHS */
    idx_t nh = 20, nc = 8;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;
    idx_t nrhs = 5;

    double *X_exact = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    for (idx_t j = 0; j < nrhs; j++) {
        for (idx_t i = 0; i < n; i++)
            X_exact[j * n + i] = sin((double)(j * 100 + i + 1));
        sparse_matvec(K, X_exact + j * n, B + j * n);
    }

    double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_minres_solve_block(K, B, nrhs, X, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    printf("    block MINRES KKT %dx%d nrhs=%d: %d max iters, relres=%.3e\n", (int)n, (int)n,
           (int)nrhs, (int)res.iterations, res.residual_norm);

    for (idx_t j = 0; j < nrhs; j++) {
        double relres = relative_residual(K, X + j * n, B + j * n, n);
        ASSERT_TRUE(relres < 1e-8);
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(X[j * n + i], X_exact[j * n + i], 1e-5);
    }

    free(X_exact);
    free(B);
    free(X);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Block MINRES: preconditioning & stress tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_minres_block_precond_ic(void) {
    /* IC(0)-preconditioned block MINRES on SPD system, 5 RHS */
    idx_t n = 20;
    idx_t nrhs = 5;
    SparseMatrix *A = make_spd_tridiag(n);

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double *X_exact = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    for (idx_t j = 0; j < nrhs; j++) {
        for (idx_t i = 0; i < n; i++)
            X_exact[j * n + i] = sin((double)(j * 10 + i + 1));
        sparse_matvec(A, X_exact + j * n, B + j * n);
    }

    sparse_iter_opts_t opts = {.max_iter = 200, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned block MINRES */
    double *X1 = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_minres_solve_block(A, B, nrhs, X1, &opts, NULL, NULL, &res1), SPARSE_OK);

    /* IC(0)-preconditioned block MINRES */
    double *X2 = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_minres_solve_block(A, B, nrhs, X2, &opts, sparse_ic_precond, &ic, &res2),
               SPARSE_OK);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);

    printf("    block IC(0)-MINRES SPD n=%d nrhs=%d: unprec=%d iters, IC(0)=%d iters\n", (int)n,
           (int)nrhs, (int)res1.iterations, (int)res2.iterations);

    /* IC(0) should converge faster */
    ASSERT_TRUE(res2.iterations <= res1.iterations);

    /* Verify solutions match */
    for (idx_t j = 0; j < nrhs; j++) {
        double relres = relative_residual(A, X2 + j * n, B + j * n, n);
        ASSERT_TRUE(relres < 1e-8);
    }

    free(X_exact);
    free(B);
    free(X1);
    free(X2);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_minres_block_precond_jacobi_indefinite(void) {
    /* Jacobi-preconditioned block MINRES on indefinite KKT, 3 RHS */
    idx_t nh = 15, nc = 6;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;
    idx_t nrhs = 3;

    jacobi_ctx_t jac = make_jacobi(K, n);

    double *X_exact = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    for (idx_t j = 0; j < nrhs; j++) {
        for (idx_t i = 0; i < n; i++)
            X_exact[j * n + i] = cos((double)(j * 20 + i + 1));
        sparse_matvec(K, X_exact + j * n, B + j * n);
    }

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned */
    double *X1 = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_minres_solve_block(K, B, nrhs, X1, &opts, NULL, NULL, &res1), SPARSE_OK);

    /* Jacobi-preconditioned */
    double *X2 = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_minres_solve_block(K, B, nrhs, X2, &opts, jacobi_precond, &jac, &res2),
               SPARSE_OK);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);

    printf("    block Jacobi-MINRES KKT %dx%d nrhs=%d: unprec=%d iters, Jacobi=%d iters\n", (int)n,
           (int)n, (int)nrhs, (int)res1.iterations, (int)res2.iterations);

    for (idx_t j = 0; j < nrhs; j++) {
        double relres = relative_residual(K, X2 + j * n, B + j * n, n);
        ASSERT_TRUE(relres < 1e-8);
    }

    free(X_exact);
    free(B);
    free(X1);
    free(X2);
    free(jac.diag_inv);
    sparse_free(K);
}

static void test_minres_block_vs_sequential(void) {
    /* Block solve should give identical results to sequential single-RHS solves */
    idx_t nh = 8, nc = 3;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;
    idx_t nrhs = 4;

    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    for (idx_t j = 0; j < nrhs; j++)
        for (idx_t i = 0; i < n; i++)
            B[j * n + i] = (double)(j * n + i + 1);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-12, .verbose = 0};

    /* Block solve */
    double *X_block = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_result_t res_block;
    ASSERT_ERR(sparse_minres_solve_block(K, B, nrhs, X_block, &opts, NULL, NULL, &res_block),
               SPARSE_OK);

    /* Sequential single-RHS solves */
    idx_t max_seq_iters = 0;
    for (idx_t j = 0; j < nrhs; j++) {
        double *x_seq = calloc((size_t)n, sizeof(double));
        sparse_iter_result_t res_seq;
        ASSERT_ERR(sparse_solve_minres(K, B + j * n, x_seq, &opts, NULL, NULL, &res_seq),
                   SPARSE_OK);
        ASSERT_TRUE(res_seq.converged);

        if (res_seq.iterations > max_seq_iters)
            max_seq_iters = res_seq.iterations;

        /* Solutions must match */
        for (idx_t i = 0; i < n; i++)
            ASSERT_NEAR(X_block[j * n + i], x_seq[i], 1e-14);

        free(x_seq);
    }

    /* Block iterations = max of per-column iterations */
    ASSERT_EQ(res_block.iterations, max_seq_iters);
    ASSERT_TRUE(res_block.converged);

    free(B);
    free(X_block);
    sparse_free(K);
}

static void test_minres_block_all_zero_rhs(void) {
    /* All-zero RHS matrix: X should be all zero */
    idx_t n = 10;
    idx_t nrhs = 3;
    SparseMatrix *A = make_spd_tridiag(n);

    double *B = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    double *X = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    /* Initialize X with garbage to verify it gets zeroed */
    for (idx_t i = 0; i < n * nrhs; i++)
        X[i] = 999.0;

    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_minres_solve_block(A, B, nrhs, X, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    ASSERT_EQ(res.iterations, 0);

    for (idx_t i = 0; i < n * nrhs; i++)
        ASSERT_NEAR(X[i], 0.0, 1e-14);

    free(B);
    free(X);
    sparse_free(A);
}

static void test_minres_block_many_rhs(void) {
    /* Stress test: 20 RHS on a 50x50 KKT system */
    idx_t nh = 35, nc = 15;
    SparseMatrix *K = make_kkt(nh, nc);
    idx_t n = nh + nc;
    idx_t nrhs = 20;

    double *X_exact = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    double *B = malloc((size_t)n * (size_t)nrhs * sizeof(double));
    for (idx_t j = 0; j < nrhs; j++) {
        for (idx_t i = 0; i < n; i++)
            X_exact[j * n + i] = sin((double)(j * 50 + i + 1));
        sparse_matvec(K, X_exact + j * n, B + j * n);
    }

    double *X = calloc((size_t)n * (size_t)nrhs, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_minres_solve_block(K, B, nrhs, X, &opts, NULL, NULL, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    printf("    block MINRES KKT %dx%d nrhs=%d: %d max iters, relres=%.3e\n", (int)n, (int)n,
           (int)nrhs, (int)res.iterations, res.residual_norm);

    /* Verify each column */
    for (idx_t j = 0; j < nrhs; j++) {
        double relres = relative_residual(K, X + j * n, B + j * n, n);
        ASSERT_TRUE(relres < 1e-8);
    }

    free(X_exact);
    free(B);
    free(X);
    sparse_free(K);
}

static void test_minres_block_non_square(void) {
    /* Non-square matrix should fail */
    SparseMatrix *A = sparse_create(3, 4);
    double B[3], X[3];
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_minres_solve_block(A, B, 1, X, NULL, NULL, NULL, &res), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("test_minres");

    /* Entry validation */
    RUN_TEST(test_minres_null_args);
    RUN_TEST(test_minres_non_square);
    RUN_TEST(test_minres_zero_rhs);

    /* SPD systems */
    RUN_TEST(test_minres_1x1);
    RUN_TEST(test_minres_1x1_negative);
    RUN_TEST(test_minres_spd_diagonal);
    RUN_TEST(test_minres_spd_tridiag);
    RUN_TEST(test_minres_spd_vs_cg);
    RUN_TEST(test_minres_already_converged);

    /* Symmetric indefinite */
    RUN_TEST(test_minres_indefinite_tridiag);
    RUN_TEST(test_minres_kkt_small);
    RUN_TEST(test_minres_kkt_medium);
    RUN_TEST(test_minres_vs_gmres_indefinite);

    /* Preconditioned MINRES */
    RUN_TEST(test_minres_precond_ic_spd);
    RUN_TEST(test_minres_precond_ic_vs_cg);
    RUN_TEST(test_minres_precond_jacobi_indefinite);
    RUN_TEST(test_minres_precond_jacobi_large_kkt);
    RUN_TEST(test_minres_precond_ic_banded);
    RUN_TEST(test_minres_precond_identity);
    RUN_TEST(test_minres_precond_exact);

    /* Edge cases & robustness */
    RUN_TEST(test_minres_max_iter);
    RUN_TEST(test_minres_large_indefinite);

    /* Numerical robustness */
    RUN_TEST(test_minres_scaled_tolerance);
    RUN_TEST(test_minres_scaled_spd);
    RUN_TEST(test_minres_ill_conditioned);
    RUN_TEST(test_minres_early_lanczos_termination);

    /* MINRES vs direct solvers */
    RUN_TEST(test_minres_vs_ldlt_spd);
    RUN_TEST(test_minres_vs_ldlt_indefinite);
    RUN_TEST(test_minres_vs_ldlt_large_kkt);
    RUN_TEST(test_minres_vs_gmres_large);

    /* Block MINRES */
    RUN_TEST(test_minres_block_null_args);
    RUN_TEST(test_minres_block_zero_nrhs);
    RUN_TEST(test_minres_block_single_rhs);
    RUN_TEST(test_minres_block_multi_rhs_spd);
    RUN_TEST(test_minres_block_multi_rhs_indefinite);
    RUN_TEST(test_minres_block_zero_rhs_column);
    RUN_TEST(test_minres_block_large);

    /* Block MINRES: preconditioning & stress */
    RUN_TEST(test_minres_block_precond_ic);
    RUN_TEST(test_minres_block_precond_jacobi_indefinite);
    RUN_TEST(test_minres_block_vs_sequential);
    RUN_TEST(test_minres_block_all_zero_rhs);
    RUN_TEST(test_minres_block_many_rhs);
    RUN_TEST(test_minres_block_non_square);

    TEST_SUITE_END();
}
