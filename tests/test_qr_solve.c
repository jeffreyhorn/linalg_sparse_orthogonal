#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include "test_qr_helpers.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

static void assert_qr_solve_reconstruction_below(const char *label, const SparseMatrix *A,
                                                 const sparse_qr_t *qr, double tol) {
    double recon_err = tf_qr_reconstruction_max_error(A, qr);
    printf("    %s: %.3e\n", label, recon_err);
    ASSERT_TRUE(recon_err < tol);
}

static double assert_qr_solve_true_residual_below(const char *label, const SparseMatrix *A,
                                                  const double *b, const double *x, idx_t m,
                                                  double reported_residual, double tol) {
    double rr = tf_qr_relative_residual_l2(A, b, x, m);
    printf("    %s: res_norm=%.3e, true_res=%.3e\n", label, reported_residual, rr);
    ASSERT_TRUE(rr < tol);
    return rr;
}

static int qr_solve_checked(const sparse_qr_t *qr, const double *b, double *x, double *residual) {
    sparse_err_t err = sparse_qr_solve(qr, b, x, residual);
    ASSERT_ERR(err, SPARSE_OK);
    return err == SPARSE_OK;
}

static void test_qr_solve_square(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    if (!tf_qr_insert_or_free(&A, 0, 0, 2.0) || !tf_qr_insert_or_free(&A, 0, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 0, 2, 1.0) || !tf_qr_insert_or_free(&A, 1, 0, 4.0) ||
        !tf_qr_insert_or_free(&A, 1, 1, 3.0) || !tf_qr_insert_or_free(&A, 1, 2, 3.0) ||
        !tf_qr_insert_or_free(&A, 2, 0, 8.0) || !tf_qr_insert_or_free(&A, 2, 1, 7.0) ||
        !tf_qr_insert_or_free(&A, 2, 2, 9.0))
        return;

    double b[3] = {1.0, 2.0, 3.0};

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    double x_qr[3];
    double res_qr;
    if (!qr_solve_checked(&qr, b, x_qr, &res_qr)) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    double x_lu[3];
    if (!LU) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_err_t lu_factor_err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
    ASSERT_ERR(lu_factor_err, SPARSE_OK);
    if (lu_factor_err != SPARSE_OK) {
        sparse_free(LU);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_err_t lu_solve_err = sparse_lu_solve(LU, b, x_lu);
    ASSERT_ERR(lu_solve_err, SPARSE_OK);
    if (lu_solve_err != SPARSE_OK) {
        sparse_free(LU);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_free(LU);

    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x_qr[i], x_lu[i], 1e-8);

    assert_qr_solve_true_residual_below("square QR solve", A, b, x_qr, 3, res_qr, 1e-10);
    ASSERT_TRUE(res_qr < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_solve_overdetermined(void) {
    SparseMatrix *A = sparse_create(5, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 1, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 2, 2, 1.0) || !tf_qr_insert_or_free(&A, 3, 0, 1.0) ||
        !tf_qr_insert_or_free(&A, 3, 1, 1.0) || !tf_qr_insert_or_free(&A, 4, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 4, 2, 1.0))
        return;

    double b[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[3];
    double res;
    if (!qr_solve_checked(&qr, b, x, &res)) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double rr = assert_qr_solve_true_residual_below("overdetermined 5x3", A, b, x, 5, res, 1.0);
    printf("    overdetermined 5x3: x=[%.3f, %.3f, %.3f]\n", x[0], x[1], x[2]);

    ASSERT_TRUE(res > 0.0);
    double bnorm = vec_norm2(b, 5);
    ASSERT_NEAR(res / bnorm, rr, 1e-8);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_solve_analytical(void) {
    SparseMatrix *A = sparse_create(2, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 1, 0, 1.0))
        return;

    double b[2] = {1.0, 3.0};

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[1];
    double res;
    if (!qr_solve_checked(&qr, b, x, &res)) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    printf("    analytical LS: x=%.6f (expected 2.0), residual=%.3e\n", x[0], res);
    ASSERT_NEAR(x[0], 2.0, 1e-10);
    ASSERT_NEAR(res, sqrt(2.0), 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_solve_overdetermined_compatible_tall(void) {
    SparseMatrix *A = sparse_create(4, 2);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 1, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 2, 0, 1.0) || !tf_qr_insert_or_free(&A, 2, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 3, 0, 2.0) || !tf_qr_insert_or_free(&A, 3, 1, -1.0))
        return;

    const double x_exact[2] = {2.0, -1.0};
    double b[4];
    sparse_matvec(A, x_exact, b);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[2];
    double res;
    if (!qr_solve_checked(&qr, b, x, &res)) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    assert_qr_solve_true_residual_below("compatible tall LS", A, b, x, 4, res, 1e-10);
    ASSERT_NEAR(x[0], x_exact[0], 1e-10);
    ASSERT_NEAR(x[1], x_exact[1], 1e-10);
    ASSERT_TRUE(res < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_solve_overdetermined_incompatible_known_residual(void) {
    SparseMatrix *A = sparse_create(4, 2);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 1, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 2, 0, 1.0) || !tf_qr_insert_or_free(&A, 2, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 3, 0, 2.0) || !tf_qr_insert_or_free(&A, 3, 1, -1.0))
        return;

    const double x_exact[2] = {2.0, -1.0};
    const double orthogonal_residual[4] = {-1.0, -1.0, 1.0, 0.0};
    double b[4];
    sparse_matvec(A, x_exact, b);
    for (idx_t i = 0; i < 4; i++)
        b[i] += orthogonal_residual[i];

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[2];
    double res;
    if (!qr_solve_checked(&qr, b, x, &res)) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double rr = assert_qr_solve_true_residual_below("incompatible tall LS", A, b, x, 4, res, 0.5);
    ASSERT_NEAR(x[0], x_exact[0], 1e-10);
    ASSERT_NEAR(x[1], x_exact[1], 1e-10);
    ASSERT_NEAR(res, sqrt(3.0), 1e-10);
    ASSERT_NEAR(res / vec_norm2(b, 4), rr, 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_solve_minnorm_underdetermined_known_solution(void) {
    SparseMatrix *A = sparse_create(2, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 0, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 1, 2, 1.0) || !tf_qr_insert_or_free(&A, 1, 3, 1.0))
        return;

    const double b[2] = {1.0, 1.0};
    double x[4];
    sparse_err_t err = sparse_qr_solve_minnorm(A, b, x, NULL);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double Ax[2];
    sparse_matvec(A, x, Ax);
    ASSERT_NEAR(Ax[0], b[0], 1e-10);
    ASSERT_NEAR(Ax[1], b[1], 1e-10);
    for (idx_t i = 0; i < 4; i++)
        ASSERT_NEAR(x[i], 0.5, 1e-10);
    ASSERT_NEAR(vec_norm2(x, 4), 1.0, 1e-10);
    printf("    minnorm underdetermined 2x4: x=[%.3f, %.3f, %.3f, %.3f]\n", x[0], x[1], x[2], x[3]);

    sparse_free(A);
}

static void test_qr_solve_rank_deficient(void) {
    SparseMatrix *A = tf_qr_make_duplicate_column_4x3(1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    double b[4] = {1.0, 2.0, 3.0, 4.0};

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 2);

    double x[3];
    double res;
    if (!qr_solve_checked(&qr, b, x, &res)) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double rr = assert_qr_solve_true_residual_below("rank-deficient solve", A, b, x, 4, res, 1.0);
    printf("    rank-deficient solve: rank=%d\n", (int)qr.rank);

    ASSERT_TRUE(rr < 1.0);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_solve_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = NULL;
    double *b = NULL;
    if (!tf_qr_make_exact_rhs(A, n, n, &x_exact, &b)) {
        sparse_free(A);
        return;
    }

    sparse_qr_t qr;
    {
        sparse_err_t ferr = sparse_qr_factor(A, &qr);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            free(x_exact);
            free(b);
            sparse_free(A);
            return;
        }
    }

    double *x_qr = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_qr);
    if (!x_qr) {
        free(x_exact);
        free(b);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    double res;
    if (!qr_solve_checked(&qr, b, x_qr, &res)) {
        free(x_exact);
        free(b);
        free(x_qr);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    printf("    nos4 QR solve: rank=%d\n", (int)qr.rank);
    assert_qr_solve_true_residual_below("nos4 QR solve", A, b, x_qr, n, res, 1e-8);

    free(x_exact);
    free(b);
    free(x_qr);
    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_solve_null_residual(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    if (!tf_qr_insert_or_free(&A, 0, 0, 2.0) || !tf_qr_insert_or_free(&A, 0, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 1, 0, 1.0) || !tf_qr_insert_or_free(&A, 1, 1, 3.0))
        return;

    double b[2] = {5.0, 5.0};

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[2];
    if (!qr_solve_checked(&qr, b, x, NULL)) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double rr = tf_qr_relative_residual_l2(A, b, x, 2);
    ASSERT_TRUE(rr < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_bcsstk04(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    sparse_qr_t qr;
    {
        sparse_err_t ferr = sparse_qr_factor(A, &qr);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    ASSERT_EQ(qr.rank, n);

    printf("    bcsstk04: rank=%d\n", (int)qr.rank);
    assert_qr_solve_reconstruction_below("bcsstk04 reconstruction", A, &qr, 1e-6);

    double *x_exact = NULL;
    double *b = NULL;
    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    if (!tf_qr_make_exact_rhs(A, n, n, &x_exact, &b) || !x) {
        free(x_exact);
        free(b);
        free(x);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double res;
    if (!qr_solve_checked(&qr, b, x, &res)) {
        free(x_exact);
        free(b);
        free(x);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    assert_qr_solve_true_residual_below("bcsstk04 QR solve", A, b, x, n, res, 1e-4);

    free(x_exact);
    free(b);
    free(x);
    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    sparse_qr_t qr;
    {
        sparse_err_t ferr = sparse_qr_factor(A, &qr);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }

    printf("    west0067: rank=%d\n", (int)qr.rank);

    double *x_exact = NULL;
    double *b = NULL;
    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    if (!tf_qr_make_exact_rhs(A, n, n, &x_exact, &b) || !x) {
        free(x_exact);
        free(b);
        free(x);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double res;
    if (!qr_solve_checked(&qr, b, x, &res)) {
        free(x_exact);
        free(b);
        free(x);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    assert_qr_solve_true_residual_below("west0067 QR solve", A, b, x, n, res, 1e-8);

    free(x_exact);
    free(b);
    free(x);
    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_vs_lu(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *x_exact = NULL;
    double *b = NULL;
    if (!tf_qr_make_exact_rhs(A, n, n, &x_exact, &b)) {
        sparse_free(A);
        return;
    }

    sparse_qr_t qr;
    {
        sparse_err_t ferr = sparse_qr_factor(A, &qr);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            free(x_exact);
            free(b);
            sparse_free(A);
            return;
        }
    }
    double *x_qr = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_qr);
    if (!x_qr || !qr_solve_checked(&qr, b, x_qr, NULL)) {
        free(x_exact);
        free(b);
        free(x_qr);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    double *x_lu = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_lu);
    if (!LU || !x_lu) {
        free(x_exact);
        free(b);
        free(x_qr);
        free(x_lu);
        sparse_free(LU);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_err_t lu_factor_err = sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
    ASSERT_ERR(lu_factor_err, SPARSE_OK);
    if (lu_factor_err != SPARSE_OK) {
        free(x_exact);
        free(b);
        free(x_qr);
        free(x_lu);
        sparse_free(LU);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_err_t lu_solve_err = sparse_lu_solve(LU, b, x_lu);
    ASSERT_ERR(lu_solve_err, SPARSE_OK);
    if (lu_solve_err != SPARSE_OK) {
        free(x_exact);
        free(b);
        free(x_qr);
        free(x_lu);
        sparse_free(LU);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double rr_qr = tf_qr_relative_residual_l2(A, b, x_qr, n);
    double rr_lu = tf_qr_relative_residual_l2(A, b, x_lu, n);
    printf("    nos4 QR vs LU: qr_res=%.3e, lu_res=%.3e\n", rr_qr, rr_lu);
    ASSERT_TRUE(rr_qr < 1e-8);
    ASSERT_TRUE(rr_lu < 1e-8);

    double maxdiff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double diff = fabs(x_qr[i] - x_lu[i]);
        if (diff > maxdiff)
            maxdiff = diff;
    }
    printf("    nos4 QR vs LU: max |diff| = %.3e\n", maxdiff);
    ASSERT_TRUE(maxdiff < 1e-4);

    free(x_exact);
    free(b);
    free(x_qr);
    free(x_lu);
    sparse_free(LU);
    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_tall_synthetic(void) {
    idx_t m = 50, nc = 20;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++) {
            double val = sin((double)(i + 1) * (double)(j + 1) * 0.3);
            if (fabs(val) > 0.25 && !tf_qr_insert_or_free(&A, i, j, val))
                return;
        }

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    printf("    50x20 synthetic: rank=%d\n", (int)qr.rank);
    ASSERT_TRUE(qr.rank <= nc);

    assert_qr_solve_reconstruction_below("50x20 reconstruction", A, &qr, 1e-10);

    double *x_exact = NULL;
    double *b = NULL;
    double *x = malloc((size_t)nc * sizeof(double));
    ASSERT_NOT_NULL(x);
    if (!tf_qr_make_exact_rhs(A, nc, m, &x_exact, &b) || !x) {
        free(x_exact);
        free(b);
        free(x);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double res;
    if (!qr_solve_checked(&qr, b, x, &res)) {
        free(x_exact);
        free(b);
        free(x);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    assert_qr_solve_true_residual_below("50x20 solve", A, b, x, m, res, 1e-8);

    free(x_exact);
    free(b);
    free(x);
    sparse_qr_free(&qr);
    sparse_free(A);
}

int main(void) {
    TEST_SUITE_BEGIN("Sparse QR solve scenario tests");

    RUN_TEST(test_qr_solve_square);
    RUN_TEST(test_qr_solve_overdetermined);
    RUN_TEST(test_qr_solve_analytical);
    RUN_TEST(test_qr_solve_overdetermined_compatible_tall);
    RUN_TEST(test_qr_solve_overdetermined_incompatible_known_residual);
    RUN_TEST(test_qr_solve_minnorm_underdetermined_known_solution);
    RUN_TEST(test_qr_solve_rank_deficient);
    RUN_TEST(test_qr_solve_nos4);
    RUN_TEST(test_qr_solve_null_residual);
    RUN_TEST(test_qr_bcsstk04);
    RUN_TEST(test_qr_west0067);
    RUN_TEST(test_qr_vs_lu);
    RUN_TEST(test_qr_tall_synthetic);

    TEST_SUITE_END();
}
