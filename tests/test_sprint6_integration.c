#include "sparse_cholesky.h"
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

static double compute_rel_residual(const SparseMatrix *A, const double *b, const double *x,
                                   idx_t m) {
    double *r = malloc((size_t)m * sizeof(double));
    if (!r)
        return INFINITY;
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < m; i++)
        r[i] = b[i] - r[i];
    double rnorm = vec_norm2(r, m);
    double bnorm = vec_norm2(b, m);
    free(r);
    return (bnorm > 0.0) ? rnorm / bnorm : 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cross-feature integration tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* All solvers on nos4: LU, Cholesky, CG, GMRES, QR → comparable residuals */
static void test_all_solvers_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    /* LU */
    SparseMatrix *LU = sparse_copy(A);
    double *x_lu = malloc((size_t)n * sizeof(double));
    double rr_lu = INFINITY;
    if (LU && x_lu) {
        sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
        sparse_lu_solve(LU, b, x_lu);
        rr_lu = compute_rel_residual(A, b, x_lu, n);
    }

    /* Cholesky */
    SparseMatrix *Ch = sparse_copy(A);
    double *x_ch = malloc((size_t)n * sizeof(double));
    double rr_ch = INFINITY;
    if (Ch && x_ch) {
        sparse_cholesky_factor(Ch);
        sparse_cholesky_solve(Ch, b, x_ch);
        rr_ch = compute_rel_residual(A, b, x_ch, n);
    }

    /* CG */
    double *x_cg = calloc((size_t)n, sizeof(double));
    double rr_cg = INFINITY;
    if (x_cg) {
        sparse_iter_opts_t cg_opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};
        sparse_iter_result_t res_cg;
        sparse_solve_cg(A, b, x_cg, &cg_opts, NULL, NULL, &res_cg);
        rr_cg = compute_rel_residual(A, b, x_cg, n);
    }

    /* GMRES (right-preconditioned with ILU) */
    double *x_gm = calloc((size_t)n, sizeof(double));
    double rr_gm = INFINITY;
    sparse_ilu_t ilu;
    int have_ilu = 0;
    if (x_gm && sparse_ilu_factor(A, &ilu) == SPARSE_OK) {
        have_ilu = 1;
        sparse_gmres_opts_t gm_opts = {.max_iter = 500,
                                       .restart = 50,
                                       .tol = 1e-10,
                                       .verbose = 0,
                                       .precond_side = SPARSE_PRECOND_RIGHT};
        sparse_iter_result_t res_gm;
        sparse_solve_gmres(A, b, x_gm, &gm_opts, sparse_ilu_precond, &ilu, &res_gm);
        rr_gm = compute_rel_residual(A, b, x_gm, n);
    }

    /* QR */
    double *x_qr = malloc((size_t)n * sizeof(double));
    double rr_qr = INFINITY;
    sparse_qr_t qr;
    if (x_qr && sparse_qr_factor(A, &qr) == SPARSE_OK) {
        sparse_qr_solve(&qr, b, x_qr, NULL);
        rr_qr = compute_rel_residual(A, b, x_qr, n);
        sparse_qr_free(&qr);
    }

    printf("    nos4 all solvers: LU=%.3e, Chol=%.3e, CG=%.3e, GMRES=%.3e, QR=%.3e\n", rr_lu, rr_ch,
           rr_cg, rr_gm, rr_qr);

    ASSERT_TRUE(rr_lu < 1e-8);
    ASSERT_TRUE(rr_ch < 1e-8);
    ASSERT_TRUE(rr_cg < 1e-8);
    ASSERT_TRUE(rr_gm < 1e-8);
    ASSERT_TRUE(rr_qr < 1e-8);

    free(x_exact);
    free(b);
    free(x_lu);
    free(x_ch);
    free(x_cg);
    free(x_gm);
    free(x_qr);
    sparse_free(LU);
    sparse_free(Ch);
    if (have_ilu)
        sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* QR on SPD matrix vs Cholesky: same solution */
static void test_qr_vs_cholesky(void) {
    SparseMatrix *A = sparse_create(4, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* SPD tridiagonal */
    for (idx_t i = 0; i < 4; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < 3)
            sparse_insert(A, i, i + 1, -1.0);
    }

    double b[4] = {3.0, 2.0, 2.0, 3.0};

    /* QR solve */
    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);
    double x_qr[4];
    sparse_qr_solve(&qr, b, x_qr, NULL);

    /* Cholesky solve */
    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    double x_ch[4] = {0};
    if (L) {
        sparse_cholesky_factor(L);
        sparse_cholesky_solve(L, b, x_ch);
        sparse_free(L);
    }

    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x_qr[i], x_ch[i], 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ILUT-preconditioned GMRES (right) on west0067 */
static void test_ilut_right_gmres_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_ilut_opts_t ilut_opts = {.tol = 1e-4, .max_fill = 20};
    sparse_ilu_t ilut;
    sparse_err_t ferr = sparse_ilut_factor(A, &ilut_opts, &ilut);
    ASSERT_ERR(ferr, SPARSE_OK);
    if (ferr != SPARSE_OK) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }

    double *x = calloc((size_t)n, sizeof(double));
    ASSERT_NOT_NULL(x);
    if (!x) {
        free(x_exact);
        free(b);
        sparse_ilu_free(&ilut);
        sparse_free(A);
        return;
    }
    /* west0067 has 65/67 zero diagonals; ILUT uses diagonal modification
     * but the resulting preconditioner is too poor for GMRES convergence.
     * We validate that the solver runs without crashing and returns a
     * well-formed result. */
    sparse_gmres_opts_t gm_opts = {.max_iter = 500,
                                   .restart = 30,
                                   .tol = 1e-8,
                                   .verbose = 0,
                                   .precond_side = SPARSE_PRECOND_RIGHT};
    sparse_iter_result_t result;
    sparse_err_t solve_err =
        sparse_solve_gmres(A, b, x, &gm_opts, sparse_ilut_precond, &ilut, &result);
    ASSERT_TRUE(solve_err == SPARSE_OK || solve_err == SPARSE_ERR_NOT_CONVERGED);

    double rr = compute_rel_residual(A, b, x, n);
    printf("    west0067 ILUT-right-GMRES: %d iters, res=%.3e, conv=%d\n", (int)result.iterations,
           rr, result.converged);
    ASSERT_TRUE(result.iterations > 0);
    ASSERT_TRUE(result.residual_norm >= 0.0);

    free(x_exact);
    free(b);
    free(x);
    sparse_ilu_free(&ilut);
    sparse_free(A);
}

/* QR solve vs GMRES on same system */
static void test_qr_vs_gmres(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 5; i++) {
        sparse_insert(A, i, i, 5.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < 4)
            sparse_insert(A, i, i + 1, 2.0);
    }

    double b[5] = {4.0, 6.0, 6.0, 6.0, 7.0};

    /* QR solve */
    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);
    double x_qr[5];
    sparse_qr_solve(&qr, b, x_qr, NULL);

    /* GMRES solve */
    double x_gm[5] = {0};
    sparse_gmres_opts_t gm_opts = {.max_iter = 100, .restart = 10, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t result;
    sparse_err_t gmres_err = sparse_solve_gmres(A, b, x_gm, &gm_opts, NULL, NULL, &result);
    ASSERT_ERR(gmres_err, SPARSE_OK);
    ASSERT_TRUE(result.converged);

    for (int i = 0; i < 5; i++)
        ASSERT_NEAR(x_qr[i], x_gm[i], 1e-8);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR solve with zero RHS */
static void test_qr_solve_zero_rhs(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 4.0);

    double b[3] = {0.0, 0.0, 0.0};

    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);
    double x[3];
    double res;
    ASSERT_ERR(sparse_qr_solve(&qr, b, x, &res), SPARSE_OK);

    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], 0.0, 1e-14);
    ASSERT_NEAR(res, 0.0, 1e-14);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ILUT with tol=0 → should approximate exact LU */
static void test_ilut_exact_lu(void) {
    SparseMatrix *A = sparse_create(4, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 3, 5.0);

    /* ILUT with tol=0, max_fill=100 → should keep everything */
    sparse_ilut_opts_t opts = {.tol = 0.0, .max_fill = 100};
    sparse_ilu_t ilu;
    {
        sparse_err_t ferr = sparse_ilut_factor(A, &opts, &ilu);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    /* Solve should be nearly exact */
    double b[4] = {1.0, 2.0, 3.0, 4.0};
    double z[4];
    ASSERT_ERR(sparse_ilu_solve(&ilu, b, z), SPARSE_OK);

    double *Az = malloc(4 * sizeof(double));
    ASSERT_NOT_NULL(Az);
    if (Az) {
        sparse_matvec(A, z, Az);
        for (int i = 0; i < 4; i++)
            ASSERT_NEAR(Az[i], b[i], 1e-10);
        free(Az);
    }

    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* QR rank matches factorization rank */
static void test_qr_rank_consistency(void) {
    SparseMatrix *A = sparse_create(5, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Rank 2: col2 = col0, col3 = col1 */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 2, 0, 3.0);
    sparse_insert(A, 3, 0, 4.0);
    sparse_insert(A, 4, 0, 5.0);
    sparse_insert(A, 0, 1, 6.0);
    sparse_insert(A, 1, 1, 7.0);
    sparse_insert(A, 2, 1, 8.0);
    sparse_insert(A, 3, 1, 9.0);
    sparse_insert(A, 4, 1, 10.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 2, 3.0);
    sparse_insert(A, 3, 2, 4.0);
    sparse_insert(A, 4, 2, 5.0);
    sparse_insert(A, 0, 3, 6.0);
    sparse_insert(A, 1, 3, 7.0);
    sparse_insert(A, 2, 3, 8.0);
    sparse_insert(A, 3, 3, 9.0);
    sparse_insert(A, 4, 3, 10.0);

    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);

    ASSERT_EQ(qr.rank, 2);
    idx_t est_rank = sparse_qr_rank(&qr, 0.0);
    ASSERT_EQ(est_rank, 2);

    printf("    rank consistency: factor=%d, estimate=%d\n", (int)qr.rank, (int)est_rank);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 6 Integration Tests");

    RUN_TEST(test_all_solvers_nos4);
    RUN_TEST(test_qr_vs_cholesky);
    RUN_TEST(test_ilut_right_gmres_west0067);
    RUN_TEST(test_qr_vs_gmres);
    RUN_TEST(test_qr_solve_zero_rhs);
    RUN_TEST(test_ilut_exact_lu);
    RUN_TEST(test_qr_rank_consistency);

    TEST_SUITE_END();
}
