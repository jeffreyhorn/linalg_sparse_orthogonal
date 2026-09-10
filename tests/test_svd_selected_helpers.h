#ifndef TEST_SVD_SELECTED_HELPERS_H
#define TEST_SVD_SELECTED_HELPERS_H

#include "sparse_qr.h"
#include "sparse_svd.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include "test_svd_helpers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* SVD rank: full-rank diagonal */
static inline void tf_svd_test_rank_full(void) {
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    idx_t rank;
    sparse_err_t err = sparse_svd_rank(A, 0.0, &rank);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    printf("    full-rank 5x5: rank=%d\n", (int)rank);
    ASSERT_EQ(rank, 5);

    sparse_free(A);
}

/* SVD rank: rank-deficient matrix */
static inline void tf_svd_test_rank_deficient(void) {
    /* 5x4 with col0=col1, col2=col3 -> rank 2 */
    SparseMatrix *A = tf_svd_make_rank_deficient_colpair_5x4();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    idx_t rank;
    sparse_err_t err = sparse_svd_rank(A, 0.0, &rank);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    printf("    rank-deficient 5x4: rank=%d\n", (int)rank);
    ASSERT_EQ(rank, 2);

    sparse_free(A);
}

/* SVD rank: nearly singular (large condition number) */
static inline void tf_svd_test_rank_nearly_singular(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1e-14); /* near zero but above machine eps */

    idx_t rank;
    sparse_err_t err = sparse_svd_rank(A, 0.0, &rank);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    printf("    nearly-singular 3x3: rank=%d\n", (int)rank);
    /* Default tol = eps * max(m,n) * sigma_max ~= 2.2e-16 * 3 * 1 ~= 6.6e-16
     * sigma_min = 1e-14 > tol, so rank should be 3 */
    ASSERT_EQ(rank, 3);

    /* With explicit tolerance 1e-12, rank = 2 */
    sparse_err_t err2 = sparse_svd_rank(A, 1e-12, &rank);
    ASSERT_ERR(err2, SPARSE_OK);
    if (err2 != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    printf("    nearly-singular 3x3 (tol=1e-12): rank=%d\n", (int)rank);
    ASSERT_EQ(rank, 2);

    sparse_free(A);
}

static inline void tf_svd_test_rank_diagonal_threshold_fixture(void) {
    const double diag[4] = {1.0, 1e-8, 1e-12, 0.0};
    SparseMatrix *A = tf_svd_make_diag_matrix(4, 4, diag, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    idx_t rank = -1;
    ASSERT_ERR(sparse_svd_rank(A, 1e-14, &rank), SPARSE_OK);
    ASSERT_EQ(rank, 3);
    ASSERT_ERR(sparse_svd_rank(A, 1e-10, &rank), SPARSE_OK);
    ASSERT_EQ(rank, 2);
    ASSERT_ERR(sparse_svd_rank(A, 1e-6, &rank), SPARSE_OK);
    ASSERT_EQ(rank, 1);
    printf("    SVD diag threshold fixture: rank(1e-14)=3, rank(1e-10)=2, rank(1e-6)=1\n");

    sparse_free(A);
}

static inline void tf_svd_test_qr_rank_dependent_row_fixture(void) {
    SparseMatrix *A = tf_svd_make_dependent_row_4x3();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    idx_t svd_rank = -1;
    sparse_err_t svd_err = sparse_svd_rank(A, 1e-10, &svd_rank);
    ASSERT_ERR(svd_err, SPARSE_OK);
    if (svd_err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_qr_t qr;
    sparse_err_t qr_err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(qr_err, SPARSE_OK);
    if (qr_err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    idx_t qr_rank = sparse_qr_rank(&qr, 1e-10);

    ASSERT_EQ(svd_rank, 2);
    ASSERT_EQ(qr_rank, 2);
    ASSERT_EQ(svd_rank, qr_rank);
    printf("    dependent-row SVD/QR rank fixture: svd=%d, qr=%d\n", (int)svd_rank, (int)qr_rank);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* SVD rank: NULL inputs */
static inline void tf_svd_test_rank_null(void) {
    idx_t rank;
    ASSERT_ERR(sparse_svd_rank(NULL, 0.0, &rank), SPARSE_ERR_NULL);
}

/* Pseudoinverse: diagonal matrix */
static inline void tf_svd_test_pinv_diagonal(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 2, 5.0);

    double *pinv_data = NULL;
    sparse_err_t err = sparse_pinv(A, 0.0, &pinv_data);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(pinv_data);
    if (!pinv_data) {
        sparse_free(A);
        return;
    }

    /* pinv of diag(2,4,5) = diag(0.5, 0.25, 0.2) */
    /* pinv is 3x3 column-major: pinv[col*3 + row] */
    ASSERT_NEAR(pinv_data[0 * 3 + 0], 0.5, 1e-10);
    ASSERT_NEAR(pinv_data[1 * 3 + 1], 0.25, 1e-10);
    ASSERT_NEAR(pinv_data[2 * 3 + 2], 0.2, 1e-10);
    /* Off-diag should be ~0 */
    ASSERT_NEAR(pinv_data[0 * 3 + 1], 0.0, 1e-10);
    ASSERT_NEAR(pinv_data[1 * 3 + 0], 0.0, 1e-10);

    free(pinv_data);
    sparse_free(A);
}

/* Pseudoinverse: Moore-Penrose condition A * A^+ * A ~= A */
static inline void tf_svd_test_pinv_moore_penrose(void) {
    idx_t m = 4, nc = 3;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Non-trivial tall matrix */
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 3, 2, 2.0);

    double *pinv_data = NULL;
    sparse_err_t err = sparse_pinv(A, 0.0, &pinv_data);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(pinv_data);
    if (!pinv_data) {
        sparse_free(A);
        return;
    }

    /* pinv is ncxm column-major: pinv[col*nc + row].
     * The helper preserves the first Moore-Penrose dimensions:
     * A(mxnc) * A+(ncxm) * A(mxnc) -> A(mxnc). */
    double max_err = tf_svd_pinv_first_moore_penrose_error(A, pinv_data, m, nc);
    printf("    Moore-Penrose ||A*A^+*A - A||_max = %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-10);

    free(pinv_data);
    sparse_free(A);
}

/* Pseudoinverse: NULL inputs */
static inline void tf_svd_test_pinv_null(void) {
    double *pinv_data = NULL;
    ASSERT_ERR(sparse_pinv(NULL, 0.0, &pinv_data), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_pinv(NULL, 0.0, NULL), SPARSE_ERR_NULL);
}

/* Pseudoinverse: rectangular tall matrix; verify A * A^+ * A ~= A */
static inline void tf_svd_test_pinv_rectangular(void) {
    /* 3x2 full column rank */
    idx_t m = 3, nc = 2;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 1, 3.0);

    double *pinv_data = NULL;
    sparse_err_t err = sparse_pinv(A, 0.0, &pinv_data);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(pinv_data);
    if (!pinv_data) {
        sparse_free(A);
        return;
    }

    /* pinv is ncxm = 2x3 column-major.
     * Verify A * A^+ * A ~= A (first Moore-Penrose condition).
     * A(mxnc) * A+(ncxm) * A(mxnc) -> A(mxnc). */
    double max_err = tf_svd_pinv_first_moore_penrose_error(A, pinv_data, m, nc);
    printf("    rectangular pinv ||A*A^+*A - A||_max = %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-10);

    free(pinv_data);
    sparse_free(A);
}

static inline void tf_svd_test_pinv_underdetermined_minnorm_solution(void) {
    const idx_t m = 2, nc = 4;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    if (!tf_svd_insert_or_free(&A, 0, 0, 1.0) || !tf_svd_insert_or_free(&A, 0, 1, 1.0) ||
        !tf_svd_insert_or_free(&A, 1, 2, 1.0) || !tf_svd_insert_or_free(&A, 1, 3, 1.0))
        return;

    double *pinv_data = NULL;
    sparse_err_t err = sparse_pinv(A, 1e-12, &pinv_data);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(pinv_data);
    if (!pinv_data) {
        sparse_free(A);
        return;
    }

    double max_err = tf_svd_pinv_first_moore_penrose_error(A, pinv_data, m, nc);
    ASSERT_TRUE(max_err < 1e-10);

    const double b[2] = {1.0, 1.0};
    double x[4] = {0.0, 0.0, 0.0, 0.0};
    for (idx_t j = 0; j < nc; j++)
        for (idx_t row = 0; row < m; row++)
            x[j] += pinv_data[(size_t)row * (size_t)nc + (size_t)j] * b[row];

    for (idx_t i = 0; i < nc; i++)
        ASSERT_NEAR(x[i], 0.5, 1e-10);

    double Ax[2];
    sparse_matvec(A, x, Ax);
    ASSERT_NEAR(Ax[0], b[0], 1e-10);
    ASSERT_NEAR(Ax[1], b[1], 1e-10);
    ASSERT_NEAR(vec_norm2(x, nc), 1.0, 1e-10);
    printf("    underdetermined pinv: ||A*A^+*A - A||_max = %.3e, ||x||=%.3f\n", max_err,
           vec_norm2(x, nc));

    free(pinv_data);
    sparse_free(A);
}

/* Low-rank approximation: rank-k of diagonal */
static inline void tf_svd_test_lowrank_diagonal(void) {
    const double diag[4] = {10.0, 5.0, 2.0, 1.0};
    SparseMatrix *A = tf_svd_make_diag_matrix(4, 4, diag, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    double *lr = NULL;
    sparse_err_t err = sparse_svd_lowrank(A, 2, &lr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(lr);
    if (!lr) {
        sparse_free(A);
        return;
    }

    /* Rank-2 approx of diag(10,5,2,1) = diag(10,5,0,0) */
    /* lr is 4x4 col-major */
    ASSERT_NEAR(lr[0 * 4 + 0], 10.0, 1e-10);
    ASSERT_NEAR(lr[1 * 4 + 1], 5.0, 1e-10);
    ASSERT_NEAR(lr[2 * 4 + 2], 0.0, 1e-10);
    ASSERT_NEAR(lr[3 * 4 + 3], 0.0, 1e-10);

    /* ||A - A_k||_F = sqrt(2^2 + 1^2) = sqrt(5) */
    double frob_err = tf_svd_dense_lowrank_frobenius_error(A, lr, 4, 4, 4);
    printf("    lowrank(2) diag: ||A - A_k||_F = %.6f (expected %.6f)\n", frob_err, sqrt(5.0));
    ASSERT_NEAR(frob_err, sqrt(5.0), 1e-10);

    free(lr);
    sparse_free(A);
}

/* Low-rank: error matches theoretical bound */
static inline void tf_svd_test_lowrank_error_bound(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Tridiagonal */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0 * (double)(i + 1));
        if (i + 1 < n)
            sparse_insert(A, i, i + 1, 1.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, 1.0);
    }

    /* Get full SVD for reference */
    sparse_svd_t svd;
    sparse_err_t serr = sparse_svd_compute(A, NULL, &svd);
    ASSERT_ERR(serr, SPARSE_OK);
    if (serr != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t rank_k = 3;
    double *lr = NULL;
    sparse_err_t lrerr = sparse_svd_lowrank(A, rank_k, &lr);
    ASSERT_ERR(lrerr, SPARSE_OK);
    if (lrerr != SPARSE_OK) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }
    ASSERT_NOT_NULL(lr);
    if (!lr) {
        sparse_svd_free(&svd);
        sparse_free(A);
        return;
    }

    /* ||A - A_k||_F should = sqrt(sum_{i=k}^{n-1} sigma_i^2) */
    double expected_sq = 0.0;
    for (idx_t i = rank_k; i < svd.k; i++)
        expected_sq += svd.sigma[i] * svd.sigma[i];
    double expected = sqrt(expected_sq);

    double actual = tf_svd_dense_lowrank_frobenius_error(A, lr, n, n, n);

    printf("    lowrank(%d) tridiag: ||A-A_k||_F = %.6f, expected = %.6f\n", (int)rank_k, actual,
           expected);
    ASSERT_NEAR(actual, expected, 1e-8);

    free(lr);
    sparse_svd_free(&svd);
    sparse_free(A);
}

/* Low-rank: NULL and bad args */
static inline void tf_svd_test_lowrank_errors(void) {
    double *lr = NULL;
    ASSERT_ERR(sparse_svd_lowrank(NULL, 2, &lr), SPARSE_ERR_NULL);

    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);

    ASSERT_ERR(sparse_svd_lowrank(A, 0, &lr), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_svd_lowrank(A, 4, &lr), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_svd_lowrank(A, -1, &lr), SPARSE_ERR_BADARG);

    sparse_free(A);
}

#endif /* TEST_SVD_SELECTED_HELPERS_H */
