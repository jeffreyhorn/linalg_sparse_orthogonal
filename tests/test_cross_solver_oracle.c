#include "sparse_cholesky.h"
#include "sparse_iterative.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "test_framework.h"
#include "test_solver_helpers.h"
#include <math.h>

#define PILOT_N 8

static int insert_or_free(SparseMatrix **A, idx_t row, idx_t col, double value) {
    sparse_err_t err = sparse_insert(*A, row, col, value);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(*A);
        *A = NULL;
        return 0;
    }
    return 1;
}

static SparseMatrix *build_spd_tridiag_fixture(void) {
    SparseMatrix *A = sparse_create(PILOT_N, PILOT_N);
    if (!A)
        return NULL;

    for (idx_t i = 0; i < PILOT_N; i++) {
        if (!insert_or_free(&A, i, i, 4.0))
            return NULL;
        if (i > 0 && !insert_or_free(&A, i, i - 1, -1.0))
            return NULL;
        if (i < PILOT_N - 1 && !insert_or_free(&A, i, i + 1, -1.0))
            return NULL;
    }
    return A;
}

static void build_exact_solution(double x_exact[PILOT_N]) {
    for (idx_t i = 0; i < PILOT_N; i++)
        x_exact[i] = 1.0 + 0.25 * (double)i;
}

static void compute_rhs(const SparseMatrix *A, const double x_exact[PILOT_N], double b[PILOT_N]) {
    for (idx_t i = 0; i < PILOT_N; i++) {
        b[i] = 0.0;
        for (idx_t j = 0; j < PILOT_N; j++)
            b[i] += sparse_get(A, i, j) * x_exact[j];
    }
}

static double max_abs_diff(const double *x, const double *y) {
    double max_diff = 0.0;
    for (idx_t i = 0; i < PILOT_N; i++) {
        double diff = fabs(x[i] - y[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

static void assert_solver_solution(const char *name, const SparseMatrix *A, const double *b,
                                   const double *x_exact, const double *x) {
    double residual = tf_relative_residual_l2(A, b, x, PILOT_N, HUGE_VAL);
    double diff = max_abs_diff(x, x_exact);
    printf("    %s: rel_res=%.3e, max|x-x_exact|=%.3e\n", name, residual, diff);
    ASSERT_TRUE(residual < 1e-10);
    ASSERT_TRUE(diff < 1e-8);
}

static void test_spd_generated_rhs_lu_chol_qr_cg_agree(void) {
    SparseMatrix *A = build_spd_tridiag_fixture();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    double x_exact[PILOT_N];
    double b[PILOT_N];
    build_exact_solution(x_exact);
    compute_rhs(A, x_exact, b);

    double x_lu[PILOT_N] = {0.0};
    SparseMatrix *A_lu = sparse_copy(A);
    ASSERT_NOT_NULL(A_lu);
    if (A_lu) {
        sparse_err_t factor_err = sparse_lu_factor(A_lu, SPARSE_PIVOT_PARTIAL, 1e-12);
        ASSERT_ERR(factor_err, SPARSE_OK);
        if (factor_err == SPARSE_OK) {
            sparse_err_t solve_err = sparse_lu_solve(A_lu, b, x_lu);
            ASSERT_ERR(solve_err, SPARSE_OK);
            if (solve_err == SPARSE_OK)
                assert_solver_solution("LU", A, b, x_exact, x_lu);
        }
        sparse_free(A_lu);
    }

    double x_chol[PILOT_N] = {0.0};
    SparseMatrix *A_chol = sparse_copy(A);
    ASSERT_NOT_NULL(A_chol);
    if (A_chol) {
        sparse_err_t factor_err = sparse_cholesky_factor(A_chol);
        ASSERT_ERR(factor_err, SPARSE_OK);
        if (factor_err == SPARSE_OK) {
            sparse_err_t solve_err = sparse_cholesky_solve(A_chol, b, x_chol);
            ASSERT_ERR(solve_err, SPARSE_OK);
            if (solve_err == SPARSE_OK)
                assert_solver_solution("Cholesky", A, b, x_exact, x_chol);
        }
        sparse_free(A_chol);
    }

    double x_qr[PILOT_N] = {0.0};
    sparse_qr_t qr;
    sparse_err_t qr_err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(qr_err, SPARSE_OK);
    if (qr_err == SPARSE_OK) {
        double reported_residual = 0.0;
        sparse_err_t solve_err = sparse_qr_solve(&qr, b, x_qr, &reported_residual);
        ASSERT_ERR(solve_err, SPARSE_OK);
        if (solve_err == SPARSE_OK) {
            assert_solver_solution("QR", A, b, x_exact, x_qr);
            ASSERT_TRUE(reported_residual < 1e-10);
        }
        sparse_qr_free(&qr);
    }

    double x_cg[PILOT_N] = {0.0};
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t cg_result;
    sparse_err_t cg_err = sparse_solve_cg(A, b, x_cg, &opts, NULL, NULL, &cg_result);
    ASSERT_ERR(cg_err, SPARSE_OK);
    if (cg_err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_TRUE(cg_result.converged);
    assert_solver_solution("CG", A, b, x_exact, x_cg);

    sparse_free(A);
}

int main(void) {
    TEST_SUITE_BEGIN("Cross-solver oracle pilot");

    RUN_TEST(test_spd_generated_rhs_lu_chol_qr_cg_agree);

    TEST_SUITE_END();
}
