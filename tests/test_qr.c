#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Householder reflection tests (Day 4)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Householder on [3, 4, 0] → should zero entries below first */
static void test_householder_3vec(void) {
    /* Factor A = [[3,1],[4,3],[0,1]] — small 3×2 matrix */
    SparseMatrix *A = sparse_create(3, 2);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 4.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 1, 1.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* R should be 2×2 upper triangular */
    ASSERT_NOT_NULL(qr.R);
    ASSERT_EQ(qr.rank, 2);

    /* R(0,0) should be -||col0|| = -5 (or +5 depending on sign convention) */
    double r00 = sparse_get_phys(qr.R, 0, 0);
    ASSERT_TRUE(fabs(fabs(r00) - 5.0) < 1e-10);

    /* R(1,0) should be 0 (upper triangular) */
    ASSERT_NEAR(sparse_get_phys(qr.R, 1, 0), 0.0, 1e-10);

    /* Verify Q*R*P^T ≈ A via Q^T*A*P = R approach:
     * Apply Q^T to columns of A and check we get R */
    printf("    3x2 QR: rank=%d, R(0,0)=%.3f, R(1,1)=%.3f\n", (int)qr.rank, r00,
           sparse_get_phys(qr.R, 1, 1));

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Householder on identity column → trivial reflection */
static void test_householder_identity(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 3; i++)
        sparse_insert(A, i, i, 1.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* R should be identity (or diagonal with ±1) */
    ASSERT_EQ(qr.rank, 3);
    for (idx_t i = 0; i < 3; i++) {
        ASSERT_NEAR(fabs(sparse_get_phys(qr.R, i, i)), 1.0, 1e-10);
    }

    /* Verify Q^T*Q = I by forming Q */
    double Q[9];
    sparse_qr_form_q(&qr, Q);

    /* Check Q^T * Q = I */
    for (idx_t i = 0; i < 3; i++) {
        for (idx_t j = 0; j < 3; j++) {
            double dot = 0.0;
            for (idx_t k = 0; k < 3; k++)
                dot += Q[(size_t)i * 3 + (size_t)k] * Q[(size_t)j * 3 + (size_t)k];
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(dot, expected, 1e-10);
        }
    }

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR on zero matrix → rank 0 */
static void test_qr_zero_matrix(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* All zeros — no insertions */

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 0);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR on 1×1 matrix */
static void test_qr_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 7.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 1);
    ASSERT_NEAR(fabs(sparse_get_phys(qr.R, 0, 0)), 7.0, 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR reconstruction: verify ||A - Q*R*P^T|| is small */
static void test_qr_reconstruction(void) {
    /* A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]] — full rank 3×3 */
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 0, 4.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 1, 2, 6.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 1, 8.0);
    sparse_insert(A, 2, 2, 10.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 3);

    /* Form Q explicitly */
    idx_t m = qr.m;
    idx_t n_cols = qr.n;
    double *Q = malloc((size_t)m * (size_t)m * sizeof(double));
    ASSERT_NOT_NULL(Q);
    if (!Q) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_qr_form_q(&qr, Q);

    /* Compute Q*R */
    /* R is stored as sparse min(m,n) × n */
    idx_t rrows = sparse_rows(qr.R);
    double maxerr = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t jp = 0; jp < n_cols; jp++) {
            /* (Q*R)(i, jp) = sum_k Q(i,k) * R(k, jp) */
            double qr_val = 0.0;
            for (idx_t kk = 0; kk < rrows; kk++) {
                double q_ik = Q[(size_t)kk * (size_t)m + (size_t)i]; /* col-major */
                double r_kj = sparse_get_phys(qr.R, kk, jp);
                qr_val += q_ik * r_kj;
            }

            /* Unpermute: A(i, col_perm[jp]) should equal (Q*R)(i, jp) */
            idx_t orig_col = qr.col_perm[jp];
            double a_val = sparse_get_phys(A, i, orig_col);
            double diff = fabs(qr_val - a_val);
            if (diff > maxerr)
                maxerr = diff;
        }
    }

    printf("    3x3 reconstruction: ||A - Q*R*P^T|| = %.3e\n", maxerr);
    ASSERT_TRUE(maxerr < 1e-10);

    free(Q);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR on rectangular tall matrix (5×3) */
static void test_qr_tall(void) {
    SparseMatrix *A = sparse_create(5, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Random-ish full-rank tall matrix */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 4.0);
    sparse_insert(A, 2, 2, 2.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 4, 2, 3.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.m, 5);
    ASSERT_EQ(qr.n, 3);
    ASSERT_EQ(qr.rank, 3); /* should be full rank */

    /* Verify Q^T*Q = I (first 3 columns) via apply_q */
    for (idx_t j = 0; j < 3; j++) {
        double ej[5] = {0};
        ej[j] = 1.0;
        double qj[5];
        sparse_qr_apply_q(&qr, 0, ej, qj);

        for (idx_t k = 0; k <= j; k++) {
            double ek[5] = {0};
            ek[k] = 1.0;
            double qk[5];
            sparse_qr_apply_q(&qr, 0, ek, qk);

            double dot = vec_dot(qj, qk, 5);
            double expected = (j == k) ? 1.0 : 0.0;
            ASSERT_NEAR(dot, expected, 1e-10);
        }
    }

    printf("    5x3 tall QR: rank=%d\n", (int)qr.rank);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR NULL inputs */
static void test_qr_null(void) {
    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(NULL, &qr), SPARSE_ERR_NULL);
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_NOT_NULL(A);
    if (A) {
        sparse_insert(A, 0, 0, 1.0);
        sparse_insert(A, 1, 1, 1.0);
        ASSERT_ERR(sparse_qr_factor(A, NULL), SPARSE_ERR_NULL);
        sparse_free(A);
    }
    /* Double free should be safe */
    memset(&qr, 0, sizeof(qr));
    sparse_qr_free(&qr);
}

/* Q application: Q*Q^T*x = x */
static void test_q_roundtrip(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 0, 1.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4], z[4];

    /* y = Q^T * x */
    sparse_qr_apply_q(&qr, 1, x, y);
    /* z = Q * y = Q * Q^T * x = x */
    sparse_qr_apply_q(&qr, 0, y, z);

    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(z[i], x[i], 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Column pivoting and extended factorization tests (Day 5)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Verify R diagonal is in decreasing magnitude (column pivoting property) */
static void test_qr_pivot_ordering(void) {
    SparseMatrix *A = sparse_create(4, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Matrix with columns of very different norms */
    sparse_insert(A, 0, 0, 0.001);
    sparse_insert(A, 1, 1, 100.0);
    sparse_insert(A, 2, 2, 10.0);
    sparse_insert(A, 3, 3, 1.0);
    /* Off-diagonal to make it interesting */
    sparse_insert(A, 0, 1, 0.5);
    sparse_insert(A, 1, 2, 0.3);
    sparse_insert(A, 2, 3, 0.7);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 4);

    /* R diagonals should be in decreasing magnitude */
    for (idx_t i = 0; i < qr.rank - 1; i++) {
        double ri = fabs(sparse_get_phys(qr.R, i, i));
        double ri1 = fabs(sparse_get_phys(qr.R, i + 1, i + 1));
        ASSERT_TRUE(ri >= ri1 - 1e-10);
    }

    /* Column permutation should be valid (each index 0..n-1 appears once) */
    int seen[4] = {0};
    for (idx_t i = 0; i < 4; i++) {
        ASSERT_TRUE(qr.col_perm[i] >= 0 && qr.col_perm[i] < 4);
        seen[qr.col_perm[i]] = 1;
    }
    for (int i = 0; i < 4; i++)
        ASSERT_TRUE(seen[i]);

    printf("    pivot ordering: R diag = [%.3f, %.3f, %.3f, %.3f]\n",
           fabs(sparse_get_phys(qr.R, 0, 0)), fabs(sparse_get_phys(qr.R, 1, 1)),
           fabs(sparse_get_phys(qr.R, 2, 2)), fabs(sparse_get_phys(qr.R, 3, 3)));

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* R should be strictly upper triangular (zeros below diagonal) */
static void test_qr_upper_triangular(void) {
    SparseMatrix *A = sparse_create(5, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 4.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 1, 6.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 2, 7.0);
    sparse_insert(A, 3, 3, 3.0);
    sparse_insert(A, 4, 0, 1.0);
    sparse_insert(A, 4, 3, 2.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Verify R is upper triangular: R(i,j) = 0 for i > j */
    idx_t rrows = sparse_rows(qr.R);
    idx_t rcols = sparse_cols(qr.R);
    for (idx_t i = 0; i < rrows; i++) {
        for (idx_t j = 0; j < i && j < rcols; j++) {
            ASSERT_NEAR(sparse_get_phys(qr.R, i, j), 0.0, 1e-12);
        }
    }

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Wide matrix (3×5): QR should work, rank ≤ 3 */
static void test_qr_wide(void) {
    SparseMatrix *A = sparse_create(3, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 0, 4, 2.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 2, 0, 2.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 2, 4, 5.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.m, 3);
    ASSERT_EQ(qr.n, 5);
    ASSERT_TRUE(qr.rank <= 3);
    printf("    3x5 wide QR: rank=%d\n", (int)qr.rank);

    /* Verify reconstruction: ||A - Q*R*P^T|| */
    double *Q = malloc(3 * 3 * sizeof(double));
    ASSERT_NOT_NULL(Q);
    if (!Q) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_qr_form_q(&qr, Q);

    idx_t rrows = sparse_rows(qr.R);
    double maxerr = 0.0;
    for (idx_t i = 0; i < 3; i++) {
        for (idx_t jp = 0; jp < 5; jp++) {
            double qr_val = 0.0;
            for (idx_t kk = 0; kk < rrows; kk++) {
                double q_ik = Q[(size_t)kk * 3 + (size_t)i];
                double r_kj = sparse_get_phys(qr.R, kk, jp);
                qr_val += q_ik * r_kj;
            }
            idx_t orig_col = qr.col_perm[jp];
            double a_val = sparse_get_phys(A, i, orig_col);
            double diff = fabs(qr_val - a_val);
            if (diff > maxerr)
                maxerr = diff;
        }
    }
    printf("    3x5 reconstruction: ||A - Q*R*P^T|| = %.3e\n", maxerr);
    ASSERT_TRUE(maxerr < 1e-10);

    free(Q);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Rank-deficient: duplicate columns → rank < n */
static void test_qr_rank_deficient(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Column 2 = 2 * Column 0, so rank should be 2 */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 2, 0, 3.0);
    sparse_insert(A, 3, 0, 4.0);

    sparse_insert(A, 0, 1, 5.0);
    sparse_insert(A, 1, 1, 6.0);
    sparse_insert(A, 2, 1, 7.0);
    sparse_insert(A, 3, 1, 8.0);

    sparse_insert(A, 0, 2, 2.0); /* 2 * col0 */
    sparse_insert(A, 1, 2, 4.0);
    sparse_insert(A, 2, 2, 6.0);
    sparse_insert(A, 3, 2, 8.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    printf("    rank-deficient 4x3: rank=%d (expected 2)\n", (int)qr.rank);
    ASSERT_EQ(qr.rank, 2);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Large-ish reconstruction test: 10×8 random-ish matrix */
static void test_qr_reconstruction_large(void) {
    idx_t m = 10, n_cols = 8;
    SparseMatrix *A = sparse_create(m, n_cols);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Fill with structured entries */
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < n_cols; j++) {
            double val = sin((double)(i + 1) * (double)(j + 1) * 0.7);
            if (fabs(val) > 0.3) /* keep ~60% entries */
                sparse_insert(A, i, j, val);
        }
    }

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    printf("    10x8 QR: rank=%d\n", (int)qr.rank);

    /* Verify reconstruction */
    double *Q = malloc((size_t)m * (size_t)m * sizeof(double));
    ASSERT_NOT_NULL(Q);
    if (!Q) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_qr_form_q(&qr, Q);

    idx_t rrows = sparse_rows(qr.R);
    double maxerr = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t jp = 0; jp < n_cols; jp++) {
            double qr_val = 0.0;
            for (idx_t kk = 0; kk < rrows; kk++) {
                double q_ik = Q[(size_t)kk * (size_t)m + (size_t)i];
                double r_kj = sparse_get_phys(qr.R, kk, jp);
                qr_val += q_ik * r_kj;
            }
            idx_t orig_col = qr.col_perm[jp];
            double a_val = sparse_get_phys(A, i, orig_col);
            double diff = fabs(qr_val - a_val);
            if (diff > maxerr)
                maxerr = diff;
        }
    }
    printf("    10x8 reconstruction: ||A - Q*R*P^T|| = %.3e\n", maxerr);
    ASSERT_TRUE(maxerr < 1e-10);

    /* Verify R diagonal ordering */
    for (idx_t i = 0; i < qr.rank - 1; i++) {
        double ri = fabs(sparse_get_phys(qr.R, i, i));
        double ri1 = fabs(sparse_get_phys(qr.R, i + 1, i + 1));
        ASSERT_TRUE(ri >= ri1 - 1e-10);
    }

    free(Q);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test suite
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sparse QR Factorization");

    /* Householder and basic QR (Day 4) */
    RUN_TEST(test_householder_3vec);
    RUN_TEST(test_householder_identity);
    RUN_TEST(test_qr_zero_matrix);
    RUN_TEST(test_qr_1x1);
    RUN_TEST(test_qr_reconstruction);
    RUN_TEST(test_qr_tall);
    RUN_TEST(test_qr_null);
    RUN_TEST(test_q_roundtrip);

    /* Column pivoting and extended tests (Day 5) */
    RUN_TEST(test_qr_pivot_ordering);
    RUN_TEST(test_qr_upper_triangular);
    RUN_TEST(test_qr_wide);
    RUN_TEST(test_qr_rank_deficient);
    RUN_TEST(test_qr_reconstruction_large);

    TEST_SUITE_END();
}
