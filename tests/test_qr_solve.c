#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

static int qr_solve_idx_count_bytes(idx_t count, size_t elem_size, size_t *bytes) {
    if (!bytes || count < 0 || elem_size == 0)
        return 0;
    if ((uintmax_t)count > (uintmax_t)SIZE_MAX)
        return 0;

    size_t count_size = (size_t)count;
    if (count_size > SIZE_MAX / elem_size)
        return 0;

    *bytes = count_size * elem_size;
    return 1;
}

static int make_qr_solve_exact_rhs(const SparseMatrix *A, idx_t x_len, idx_t b_len,
                                   double **x_exact_out, double **b_out) {
    if (!x_exact_out || !b_out)
        return 0;
    *x_exact_out = NULL;
    *b_out = NULL;
    if (!A) {
        ASSERT_NOT_NULL(A);
        return 0;
    }
    if (x_len != sparse_cols(A)) {
        ASSERT_EQ(x_len, sparse_cols(A));
        return 0;
    }
    if (b_len != sparse_rows(A)) {
        ASSERT_EQ(b_len, sparse_rows(A));
        return 0;
    }

    size_t x_bytes = 0;
    size_t b_bytes = 0;
    if (!qr_solve_idx_count_bytes(x_len, sizeof(double), &x_bytes) ||
        !qr_solve_idx_count_bytes(b_len, sizeof(double), &b_bytes)) {
        ASSERT_TRUE(0);
        return 0;
    }

    double *x_exact = malloc(x_bytes);
    double *b = malloc(b_bytes);
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (!x_exact || !b) {
        free(x_exact);
        free(b);
        return 0;
    }

    for (idx_t i = 0; i < x_len; i++)
        x_exact[i] = (double)(i + 1);
    sparse_err_t mv_err = sparse_matvec(A, x_exact, b);
    ASSERT_ERR(mv_err, SPARSE_OK);
    if (mv_err != SPARSE_OK) {
        free(x_exact);
        free(b);
        return 0;
    }

    *x_exact_out = x_exact;
    *b_out = b;
    return 1;
}

static int qr_solve_insert_or_free(SparseMatrix **A, idx_t row, idx_t col, double value) {
    sparse_err_t err = sparse_insert(*A, row, col, value);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(*A);
        *A = NULL;
        return 0;
    }
    return 1;
}

static SparseMatrix *make_qr_solve_duplicate_column_4x3(double duplicate_scale) {
    SparseMatrix *A = sparse_create(4, 3);
    if (!A)
        return NULL;
    if (!qr_solve_insert_or_free(&A, 0, 0, 1.0) || !qr_solve_insert_or_free(&A, 1, 0, 2.0) ||
        !qr_solve_insert_or_free(&A, 2, 0, 3.0) || !qr_solve_insert_or_free(&A, 3, 0, 4.0) ||
        !qr_solve_insert_or_free(&A, 0, 1, 5.0) || !qr_solve_insert_or_free(&A, 1, 1, 6.0) ||
        !qr_solve_insert_or_free(&A, 2, 1, 7.0) || !qr_solve_insert_or_free(&A, 3, 1, 8.0) ||
        !qr_solve_insert_or_free(&A, 0, 2, duplicate_scale) ||
        !qr_solve_insert_or_free(&A, 1, 2, duplicate_scale * 2.0) ||
        !qr_solve_insert_or_free(&A, 2, 2, duplicate_scale * 3.0) ||
        !qr_solve_insert_or_free(&A, 3, 2, duplicate_scale * 4.0))
        return NULL;
    return A;
}

static double qr_solve_reconstruction_error(const SparseMatrix *A, const sparse_qr_t *qr) {
    idx_t m = qr->m;
    idx_t n_cols = qr->n;
    size_t m_size = 0;
    if (m < 0 || (uintmax_t)m > (uintmax_t)SIZE_MAX) {
        ASSERT_TRUE(0);
        return HUGE_VAL;
    }
    m_size = (size_t)m;
    if (m_size != 0 && m_size > SIZE_MAX / m_size) {
        ASSERT_TRUE(0);
        return HUGE_VAL;
    }
    size_t q_count = m_size * m_size;
    size_t q_bytes = 0;
    if (q_count > SIZE_MAX / sizeof(double)) {
        ASSERT_TRUE(0);
        return HUGE_VAL;
    }
    q_bytes = q_count * sizeof(double);

    double *Q = malloc(q_bytes);
    if (!Q)
        return HUGE_VAL;
    sparse_err_t q_err = sparse_qr_form_q(qr, Q);
    ASSERT_ERR(q_err, SPARSE_OK);
    if (q_err != SPARSE_OK) {
        free(Q);
        return HUGE_VAL;
    }

    idx_t rrows = sparse_rows(qr->R);
    double maxerr = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t jp = 0; jp < n_cols; jp++) {
            double qr_val = 0.0;
            for (idx_t kk = 0; kk < rrows; kk++) {
                double q_ik = Q[(size_t)kk * (size_t)m + (size_t)i];
                double r_kj = sparse_get_phys(qr->R, kk, jp);
                qr_val += q_ik * r_kj;
            }
            idx_t orig_col = qr->col_perm[jp];
            double a_val = sparse_get_phys(A, i, orig_col);
            double diff = fabs(qr_val - a_val);
            if (diff > maxerr)
                maxerr = diff;
        }
    }
    free(Q);
    return maxerr;
}

static void assert_qr_solve_reconstruction_below(const char *label, const SparseMatrix *A,
                                                 const sparse_qr_t *qr, double tol) {
    double recon_err = qr_solve_reconstruction_error(A, qr);
    printf("    %s: %.3e\n", label, recon_err);
    ASSERT_TRUE(recon_err < tol);
}

static double qr_solve_rel_residual(const SparseMatrix *A, const double *b, const double *x,
                                    idx_t m) {
    double *r = malloc((size_t)m * sizeof(double));
    if (!r)
        return HUGE_VAL;
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < m; i++)
        r[i] = b[i] - r[i];
    double rnorm = vec_norm2(r, m);
    double bnorm = vec_norm2(b, m);
    free(r);
    return (bnorm > 0.0) ? rnorm / bnorm : 0.0;
}

static double assert_qr_solve_true_residual_below(const char *label, const SparseMatrix *A,
                                                  const double *b, const double *x, idx_t m,
                                                  double reported_residual, double tol) {
    double rr = qr_solve_rel_residual(A, b, x, m);
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
    if (!qr_solve_insert_or_free(&A, 0, 0, 2.0) || !qr_solve_insert_or_free(&A, 0, 1, 1.0) ||
        !qr_solve_insert_or_free(&A, 0, 2, 1.0) || !qr_solve_insert_or_free(&A, 1, 0, 4.0) ||
        !qr_solve_insert_or_free(&A, 1, 1, 3.0) || !qr_solve_insert_or_free(&A, 1, 2, 3.0) ||
        !qr_solve_insert_or_free(&A, 2, 0, 8.0) || !qr_solve_insert_or_free(&A, 2, 1, 7.0) ||
        !qr_solve_insert_or_free(&A, 2, 2, 9.0))
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
    if (LU) {
        ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
        ASSERT_ERR(sparse_lu_solve(LU, b, x_lu), SPARSE_OK);
        sparse_free(LU);

        for (int i = 0; i < 3; i++)
            ASSERT_NEAR(x_qr[i], x_lu[i], 1e-8);
    }

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
    if (!qr_solve_insert_or_free(&A, 0, 0, 1.0) || !qr_solve_insert_or_free(&A, 1, 1, 1.0) ||
        !qr_solve_insert_or_free(&A, 2, 2, 1.0) || !qr_solve_insert_or_free(&A, 3, 0, 1.0) ||
        !qr_solve_insert_or_free(&A, 3, 1, 1.0) || !qr_solve_insert_or_free(&A, 4, 1, 1.0) ||
        !qr_solve_insert_or_free(&A, 4, 2, 1.0))
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
    if (!qr_solve_insert_or_free(&A, 0, 0, 1.0) || !qr_solve_insert_or_free(&A, 1, 0, 1.0))
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

static void test_qr_solve_rank_deficient(void) {
    SparseMatrix *A = make_qr_solve_duplicate_column_4x3(1.0);
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
    if (!make_qr_solve_exact_rhs(A, n, n, &x_exact, &b)) {
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
    if (!qr_solve_insert_or_free(&A, 0, 0, 2.0) || !qr_solve_insert_or_free(&A, 0, 1, 1.0) ||
        !qr_solve_insert_or_free(&A, 1, 0, 1.0) || !qr_solve_insert_or_free(&A, 1, 1, 3.0))
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

    double rr = qr_solve_rel_residual(A, b, x, 2);
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
    if (!make_qr_solve_exact_rhs(A, n, n, &x_exact, &b) || !x) {
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
    if (!make_qr_solve_exact_rhs(A, n, n, &x_exact, &b) || !x) {
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
    if (!make_qr_solve_exact_rhs(A, n, n, &x_exact, &b)) {
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
    if (x_qr && !qr_solve_checked(&qr, b, x_qr, NULL)) {
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
    if (LU && x_lu) {
        sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
        sparse_lu_solve(LU, b, x_lu);
    }

    if (x_qr && x_lu) {
        double rr_qr = qr_solve_rel_residual(A, b, x_qr, n);
        double rr_lu = qr_solve_rel_residual(A, b, x_lu, n);
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
    }

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
            if (fabs(val) > 0.25 && !qr_solve_insert_or_free(&A, i, j, val))
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
    if (!make_qr_solve_exact_rhs(A, nc, m, &x_exact, &b) || !x) {
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
    RUN_TEST(test_qr_solve_rank_deficient);
    RUN_TEST(test_qr_solve_nos4);
    RUN_TEST(test_qr_solve_null_residual);
    RUN_TEST(test_qr_bcsstk04);
    RUN_TEST(test_qr_west0067);
    RUN_TEST(test_qr_vs_lu);
    RUN_TEST(test_qr_tall_synthetic);

    TEST_SUITE_END();
}
