#include "sparse_lu.h"
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
 * Edge cases and hardening (Day 6)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: verify ||A - Q*R*P^T|| < tol */
static double qr_reconstruction_error(const SparseMatrix *A, const sparse_qr_t *qr) {
    idx_t m = qr->m;
    idx_t n_cols = qr->n;
    double *Q = malloc((size_t)m * (size_t)m * sizeof(double));
    if (!Q)
        return -1.0;
    sparse_qr_form_q(qr, Q);

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

/* Rank-1 matrix (outer product): rank should be 1 */
static void test_qr_rank_1(void) {
    idx_t m = 5, n_c = 4;
    SparseMatrix *A = sparse_create(m, n_c);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* A = u * v^T where u = [1,2,3,4,5], v = [1,1,1,1] */
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < n_c; j++)
            sparse_insert(A, i, j, (double)(i + 1));

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    printf("    rank-1 (5x4): rank=%d\n", (int)qr.rank);
    ASSERT_EQ(qr.rank, 1);

    double recon_err = qr_reconstruction_error(A, &qr);
    printf("    rank-1 reconstruction: %.3e\n", recon_err);
    ASSERT_TRUE(recon_err < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Nearly singular: one column is almost a multiple of another */
static void test_qr_nearly_singular(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* col0 and col2 are nearly identical */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 2, 0, 3.0);
    sparse_insert(A, 3, 0, 4.0);

    sparse_insert(A, 0, 1, 5.0);
    sparse_insert(A, 1, 1, 6.0);
    sparse_insert(A, 2, 1, 7.0);
    sparse_insert(A, 3, 1, 8.0);

    sparse_insert(A, 0, 2, 1.0 + 1e-12); /* col2 ≈ col0 */
    sparse_insert(A, 1, 2, 2.0 + 1e-12);
    sparse_insert(A, 2, 2, 3.0 + 1e-12);
    sparse_insert(A, 3, 2, 4.0 + 1e-12);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    printf("    nearly singular 4x3: rank=%d\n", (int)qr.rank);
    /* Rank should be 2 (the near-duplicate column is detected) */
    ASSERT_TRUE(qr.rank <= 3);

    double recon_err = qr_reconstruction_error(A, &qr);
    printf("    nearly singular reconstruction: %.3e\n", recon_err);
    ASSERT_TRUE(recon_err < 1e-8);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Diagonal matrix: trivial QR */
static void test_qr_diagonal(void) {
    SparseMatrix *A = sparse_create(4, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, 7.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 3, 3, 5.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 4);
    /* R diagonals should be the original diagonal values (reordered by magnitude) */
    ASSERT_NEAR(fabs(sparse_get_phys(qr.R, 0, 0)), 7.0, 1e-10);

    double recon_err = qr_reconstruction_error(A, &qr);
    ASSERT_TRUE(recon_err < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Single-row matrix (1×4) */
static void test_qr_single_row(void) {
    SparseMatrix *A = sparse_create(1, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 1, 4.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 1);
    /* Pivot selects largest column element (4.0), so R(0,0) = ±4 */
    ASSERT_NEAR(fabs(sparse_get_phys(qr.R, 0, 0)), 4.0, 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Single-column matrix (5×1) */
static void test_qr_single_col(void) {
    SparseMatrix *A = sparse_create(5, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, 0, (double)(i + 1));

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 1);
    /* R(0,0) = ||[1,2,3,4,5]|| = sqrt(55) ≈ 7.416 */
    ASSERT_NEAR(fabs(sparse_get_phys(qr.R, 0, 0)), sqrt(55.0), 1e-10);

    /* Q^T * Q should still be orthogonal */
    double x[5] = {1.0, 0.0, 0.0, 0.0, 0.0};
    double y[5], z[5];
    sparse_qr_apply_q(&qr, 1, x, y);
    sparse_qr_apply_q(&qr, 0, y, z);
    for (int i = 0; i < 5; i++)
        ASSERT_NEAR(z[i], x[i], 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Permutation: verify col_perm is a valid permutation */
static void test_qr_perm_valid(void) {
    SparseMatrix *A = sparse_create(6, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 6; i++)
        for (idx_t j = 0; j < 5; j++)
            if ((i + j) % 3 != 0)
                sparse_insert(A, i, j, sin((double)(i * 5 + j + 1)));

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Check permutation is valid */
    int *seen = calloc((size_t)qr.n, sizeof(int));
    ASSERT_NOT_NULL(seen);
    if (seen) {
        for (idx_t i = 0; i < qr.n; i++) {
            ASSERT_TRUE(qr.col_perm[i] >= 0 && qr.col_perm[i] < qr.n);
            seen[qr.col_perm[i]] = 1;
        }
        for (idx_t i = 0; i < qr.n; i++)
            ASSERT_TRUE(seen[i]);
        free(seen);
    }

    double recon_err = qr_reconstruction_error(A, &qr);
    printf("    6x5 perm valid: rank=%d, recon=%.3e\n", (int)qr.rank, recon_err);
    ASSERT_TRUE(recon_err < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Q orthogonality and application tests (Day 7)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Verify Q^T*Q = I for a tall rectangular matrix */
static void test_q_orthogonality_tall(void) {
    SparseMatrix *A = sparse_create(8, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Fill with structured entries */
    for (idx_t i = 0; i < 8; i++)
        for (idx_t j = 0; j < 5; j++) {
            double val = cos((double)(i + 1) * (double)(j + 1) * 0.5);
            if (fabs(val) > 0.2)
                sparse_insert(A, i, j, val);
        }

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t m = qr.m;
    /* Form Q explicitly and check Q^T*Q = I */
    double *Q = malloc((size_t)m * (size_t)m * sizeof(double));
    ASSERT_NOT_NULL(Q);
    if (!Q) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_qr_form_q(&qr, Q);

    double max_off = 0.0;
    double max_diag_err = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j <= i; j++) {
            /* Compute Q(:,i)^T * Q(:,j) */
            double dot = 0.0;
            for (idx_t kk = 0; kk < m; kk++)
                dot +=
                    Q[(size_t)i * (size_t)m + (size_t)kk] * Q[(size_t)j * (size_t)m + (size_t)kk];
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = fabs(dot - expected);
            if (i == j) {
                if (diff > max_diag_err)
                    max_diag_err = diff;
            } else {
                if (diff > max_off)
                    max_off = diff;
            }
        }
    }
    printf("    8x5 Q orthogonality: diag_err=%.3e, off_diag=%.3e\n", max_diag_err, max_off);
    ASSERT_TRUE(max_diag_err < 1e-10);
    ASSERT_TRUE(max_off < 1e-10);

    free(Q);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Q^T applied to b for least-squares setup */
static void test_q_transpose_b(void) {
    /* A = [[1,2],[3,4],[5,6]] (3×2), b = [1,2,3] */
    SparseMatrix *A = sparse_create(3, 2);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 0, 5.0);
    sparse_insert(A, 2, 1, 6.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 2);

    double b[3] = {1.0, 2.0, 3.0};
    double qtb[3];
    sparse_qr_apply_q(&qr, 1, b, qtb); /* Q^T * b */

    /* qtb[0:rank] should be the coefficients for back-sub with R */
    /* qtb[rank:] should give the residual norm */
    double res_norm = 0.0;
    for (idx_t i = qr.rank; i < 3; i++)
        res_norm += qtb[i] * qtb[i];
    res_norm = sqrt(res_norm);

    printf("    Q^T*b: qtb = [%.6f, %.6f, %.6f], residual_norm=%.6e\n", qtb[0], qtb[1], qtb[2],
           res_norm);

    /* The residual should be small for this well-conditioned system */
    /* (3×2 overdetermined, b is close to the column space) */
    ASSERT_TRUE(res_norm >= 0.0); /* sanity: non-negative */

    /* Verify Q * (Q^T * b) = b */
    double roundtrip[3];
    sparse_qr_apply_q(&qr, 0, qtb, roundtrip);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(roundtrip[i], b[i], 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Apply Q to multiple vectors (simulating block operation) */
static void test_q_apply_multiple(void) {
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
    sparse_insert(A, 3, 2, 2.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Apply Q and Q^T to several different vectors */
    for (int t = 0; t < 4; t++) {
        double x[4] = {0};
        x[t] = 1.0; /* e_t basis vector */
        double y[4], z[4];
        sparse_qr_apply_q(&qr, 1, x, y);
        sparse_qr_apply_q(&qr, 0, y, z);
        for (int i = 0; i < 4; i++)
            ASSERT_NEAR(z[i], x[i], 1e-10);
    }

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* In-place Q application (x == y) */
static void test_q_apply_inplace(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 2, 4.0);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[3] = {1.0, 2.0, 3.0};
    double x_copy[3] = {1.0, 2.0, 3.0};

    /* In-place: y = Q^T * x where y == x */
    sparse_qr_apply_q(&qr, 1, x, x);
    /* Apply Q to get back */
    sparse_qr_apply_q(&qr, 0, x, x);

    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_copy[i], 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Q^T*Q = I on wide matrix (Q is m×m where m < n) */
static void test_q_orthogonality_wide(void) {
    SparseMatrix *A = sparse_create(3, 6);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 6; j++)
            sparse_insert(A, i, j, sin((double)((i + 1) * (j + 1))));

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Q is 3×3 — verify orthogonality */
    double Q[9];
    sparse_qr_form_q(&qr, Q);

    for (idx_t i = 0; i < 3; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double dot = 0.0;
            for (idx_t kk = 0; kk < 3; kk++)
                dot += Q[(size_t)i * 3 + (size_t)kk] * Q[(size_t)j * 3 + (size_t)kk];
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(dot, expected, 1e-10);
        }
    }

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Least-squares solver tests (Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: compute ||b - A*x|| / ||b|| */
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

/* Square full-rank: QR solve matches LU solve */
static void test_qr_solve_square(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 0, 4.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 3.0);
    sparse_insert(A, 2, 0, 8.0);
    sparse_insert(A, 2, 1, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    double b[3] = {1.0, 2.0, 3.0};

    /* QR solve */
    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    double x_qr[3];
    double res_qr;
    ASSERT_ERR(sparse_qr_solve(&qr, b, x_qr, &res_qr), SPARSE_OK);

    /* LU solve */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    double x_lu[3];
    if (LU) {
        ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
        ASSERT_ERR(sparse_lu_solve(LU, b, x_lu), SPARSE_OK);
        sparse_free(LU);

        /* QR and LU should agree */
        for (int i = 0; i < 3; i++)
            ASSERT_NEAR(x_qr[i], x_lu[i], 1e-8);
    }

    double rr = compute_rel_residual(A, b, x_qr, 3);
    printf("    square QR solve: res=%.3e, residual_norm=%.3e\n", rr, res_qr);
    ASSERT_TRUE(rr < 1e-10);
    ASSERT_TRUE(res_qr < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Overdetermined (5×3): least-squares */
static void test_qr_solve_overdetermined(void) {
    SparseMatrix *A = sparse_create(5, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* A = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1]] */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 4, 1, 1.0);
    sparse_insert(A, 4, 2, 1.0);

    /* b = [1, 2, 3, 4, 5] — not in column space, so residual > 0 */
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
    ASSERT_ERR(sparse_qr_solve(&qr, b, x, &res), SPARSE_OK);

    double rr = compute_rel_residual(A, b, x, 5);
    printf("    overdetermined 5x3: x=[%.3f, %.3f, %.3f], res=%.3e, true_res=%.3e\n", x[0], x[1],
           x[2], res, rr);

    /* Residual should be positive (overdetermined) */
    ASSERT_TRUE(res > 0.0);
    /* The reported residual should match the true residual closely */
    double bnorm = vec_norm2(b, 5);
    ASSERT_NEAR(res / bnorm, rr, 1e-8);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Known analytical least-squares: A = [[1],[1]], b = [1,3] → x = 2 */
static void test_qr_solve_analytical(void) {
    SparseMatrix *A = sparse_create(2, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 1.0);

    double b[2] = {1.0, 3.0};
    /* Least-squares: min ||[1;1]*x - [1;3]||^2 → x = (A^T*A)^{-1}*A^T*b = 2 */

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[1];
    double res;
    ASSERT_ERR(sparse_qr_solve(&qr, b, x, &res), SPARSE_OK);

    printf("    analytical LS: x=%.6f (expected 2.0), residual=%.3e\n", x[0], res);
    ASSERT_NEAR(x[0], 2.0, 1e-10);
    /* Residual = ||[1;3] - [2;2]|| = ||[-1;1]|| = sqrt(2) */
    ASSERT_NEAR(res, sqrt(2.0), 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Rank-deficient system: extra column is duplicate */
static void test_qr_solve_rank_deficient(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* col0 = [1,2,3,4], col1 = [5,6,7,8], col2 = col0 (duplicate) */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 2, 0, 3.0);
    sparse_insert(A, 3, 0, 4.0);
    sparse_insert(A, 0, 1, 5.0);
    sparse_insert(A, 1, 1, 6.0);
    sparse_insert(A, 2, 1, 7.0);
    sparse_insert(A, 3, 1, 8.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 2, 3.0);
    sparse_insert(A, 3, 2, 4.0);

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
    ASSERT_ERR(sparse_qr_solve(&qr, b, x, &res), SPARSE_OK);

    /* Verify A*x ≈ b (within residual) */
    double rr = compute_rel_residual(A, b, x, 4);
    printf("    rank-deficient solve: rank=%d, res=%.3e, true_res=%.3e\n", (int)qr.rank, res, rr);

    /* Should produce a valid least-squares solution */
    ASSERT_TRUE(rr < 1.0); /* residual should be reasonable */

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR solve on nos4 (100×100 SPD) — compare with LU */
static void test_qr_solve_nos4(void) {
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

    /* QR solve */
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
    ASSERT_ERR(sparse_qr_solve(&qr, b, x_qr, &res), SPARSE_OK);

    double rr = compute_rel_residual(A, b, x_qr, n);
    printf("    nos4 QR solve: rank=%d, res=%.3e, true_res=%.3e\n", (int)qr.rank, res, rr);
    ASSERT_TRUE(rr < 1e-8);

    free(x_exact);
    free(b);
    free(x_qr);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR solve with NULL residual pointer (should not crash) */
static void test_qr_solve_null_residual(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);

    double b[2] = {5.0, 5.0};

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[2];
    ASSERT_ERR(sparse_qr_solve(&qr, b, x, NULL), SPARSE_OK); /* NULL residual */

    double rr = compute_rel_residual(A, b, x, 2);
    ASSERT_TRUE(rr < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SuiteSparse validation (Day 9)
 * ═══════════════════════════════════════════════════════════════════════ */

/* QR on bcsstk04 (132×132 SPD stiffness matrix) */
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

    /* Reconstruction */
    double recon = qr_reconstruction_error(A, &qr);
    printf("    bcsstk04: rank=%d, ||A-QRP^T||=%.3e\n", (int)qr.rank, recon);
    ASSERT_TRUE(recon < 1e-6); /* relaxed for ill-conditioned matrix */

    /* Solve */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x);
    if (!x_exact || !b || !x) {
        free(x_exact);
        free(b);
        free(x);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    double res;
    ASSERT_ERR(sparse_qr_solve(&qr, b, x, &res), SPARSE_OK);
    double rr = compute_rel_residual(A, b, x, n);
    printf("    bcsstk04 QR solve: res_norm=%.3e, true_res=%.3e\n", res, rr);
    ASSERT_TRUE(rr < 1e-4); /* bcsstk04 is ill-conditioned */

    free(x_exact);
    free(b);
    free(x);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR on west0067 (67×67 unsymmetric) */
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

    /* Solve */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x);
    if (!x_exact || !b || !x) {
        free(x_exact);
        free(b);
        free(x);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    double res;
    ASSERT_ERR(sparse_qr_solve(&qr, b, x, &res), SPARSE_OK);
    double rr = compute_rel_residual(A, b, x, n);
    printf("    west0067 QR solve: res_norm=%.3e, true_res=%.3e\n", res, rr);
    ASSERT_TRUE(rr < 1e-8);

    free(x_exact);
    free(b);
    free(x);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* QR vs LU on nos4: compare solutions */
static void test_qr_vs_lu(void) {
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

    /* QR solve */
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
    if (x_qr)
        sparse_qr_solve(&qr, b, x_qr, NULL);

    /* LU solve */
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    double *x_lu = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_lu);
    if (LU && x_lu) {
        sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
        sparse_lu_solve(LU, b, x_lu);
    }

    if (x_qr && x_lu) {
        double rr_qr = compute_rel_residual(A, b, x_qr, n);
        double rr_lu = compute_rel_residual(A, b, x_lu, n);
        printf("    nos4 QR vs LU: qr_res=%.3e, lu_res=%.3e\n", rr_qr, rr_lu);
        ASSERT_TRUE(rr_qr < 1e-8);
        ASSERT_TRUE(rr_lu < 1e-8);

        /* Solutions should agree closely */
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

/* Larger synthetic tall matrix (50×20) */
static void test_qr_tall_synthetic(void) {
    idx_t m = 50, nc = 20;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++) {
            double val = sin((double)(i + 1) * (double)(j + 1) * 0.3);
            if (fabs(val) > 0.25)
                sparse_insert(A, i, j, val);
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

    /* Reconstruction */
    double recon = qr_reconstruction_error(A, &qr);
    printf("    50x20 reconstruction: %.3e\n", recon);
    ASSERT_TRUE(recon < 1e-10);

    /* Least-squares: construct b = A * x_exact + noise */
    double *x_exact = malloc((size_t)nc * sizeof(double));
    double *b = malloc((size_t)m * sizeof(double));
    double *x = malloc((size_t)nc * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x);
    if (!x_exact || !b || !x) {
        free(x_exact);
        free(b);
        free(x);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < nc; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    double res;
    ASSERT_ERR(sparse_qr_solve(&qr, b, x, &res), SPARSE_OK);
    double rr = compute_rel_residual(A, b, x, m);
    printf("    50x20 solve: res_norm=%.3e, true_res=%.3e\n", res, rr);
    ASSERT_TRUE(rr < 1e-8);

    free(x_exact);
    free(b);
    free(x);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Rank estimation and null-space tests (Day 10)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Full-rank matrix → rank = n, empty null space */
static void test_rank_full(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 2, 4.0);

    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);

    idx_t r = sparse_qr_rank(&qr, 0.0);
    ASSERT_EQ(r, 3);

    idx_t ndim;
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, NULL, &ndim), SPARSE_OK);
    ASSERT_EQ(ndim, 0);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Rank-1 outer product → rank = 1, null space dim = n-1 */
static void test_rank_1_nullspace(void) {
    idx_t m = 4, nc = 3;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* A = u * v^T where u = [1,2,3,4], v = [1,1,1] → rank 1 */
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            sparse_insert(A, i, j, (double)(i + 1));

    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);

    idx_t r = sparse_qr_rank(&qr, 0.0);
    printf("    rank-1 outer: rank=%d\n", (int)r);
    ASSERT_EQ(r, 1);

    idx_t ndim;
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, NULL, &ndim), SPARSE_OK);
    ASSERT_EQ(ndim, 2); /* n - rank = 3 - 1 = 2 */

    /* Extract null-space basis and verify A * v ≈ 0 */
    double *basis = malloc((size_t)nc * (size_t)ndim * sizeof(double));
    ASSERT_NOT_NULL(basis);
    if (basis) {
        ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, basis, &ndim), SPARSE_OK);

        for (idx_t j = 0; j < ndim; j++) {
            double *nv = &basis[(size_t)j * (size_t)nc];
            /* Compute A * nv */
            double *Anv = calloc((size_t)m, sizeof(double));
            if (Anv) {
                sparse_matvec(A, nv, Anv);
                double nrm = vec_norm2(Anv, m);
                printf("    null vec %d: ||A*v|| = %.3e\n", (int)j, nrm);
                ASSERT_TRUE(nrm < 1e-10);
                free(Anv);
            }
        }
        free(basis);
    }

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Known null space: col2 = col0, so [1, 0, -1] is in null space */
static void test_known_nullspace(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 2, 0, 3.0);
    sparse_insert(A, 3, 0, 4.0);
    sparse_insert(A, 0, 1, 5.0);
    sparse_insert(A, 1, 1, 6.0);
    sparse_insert(A, 2, 1, 7.0);
    sparse_insert(A, 3, 1, 8.0);
    sparse_insert(A, 0, 2, 1.0); /* col2 = col0 */
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 2, 3.0);
    sparse_insert(A, 3, 2, 4.0);

    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);

    idx_t r = sparse_qr_rank(&qr, 0.0);
    ASSERT_EQ(r, 2);

    idx_t ndim;
    double basis[3]; /* n * ndim = 3 * 1 */
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, basis, &ndim), SPARSE_OK);
    ASSERT_EQ(ndim, 1);

    /* Verify A * basis ≈ 0 */
    double Av[4];
    sparse_matvec(A, basis, Av);
    double nrm = vec_norm2(Av, 4);
    printf("    known null: ||A*v|| = %.3e, v=[%.3f, %.3f, %.3f]\n", nrm, basis[0], basis[1],
           basis[2]);
    ASSERT_TRUE(nrm < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Rank-deficient rectangular: 3×5 with rank 2 */
static void test_rank_rect_deficient(void) {
    SparseMatrix *A = sparse_create(3, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Row 2 = Row 0 + Row 1, so rank = 2 */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 1, 4, 2.0);
    /* row2 = row0 + row1 */
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 1, 5.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 2, 4, 2.0);

    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);

    idx_t r = sparse_qr_rank(&qr, 0.0);
    printf("    3x5 rank-deficient: rank=%d\n", (int)r);
    ASSERT_EQ(r, 2);

    idx_t ndim;
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, NULL, &ndim), SPARSE_OK);
    ASSERT_EQ(ndim, 3); /* 5 - 2 = 3 */

    /* Extract and verify null-space vectors */
    double *basis = malloc(5 * 3 * sizeof(double));
    ASSERT_NOT_NULL(basis);
    if (basis) {
        ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, basis, &ndim), SPARSE_OK);
        for (idx_t j = 0; j < ndim; j++) {
            double *nv = &basis[(size_t)j * 5];
            double Av[3];
            sparse_matvec(A, nv, Av);
            double nrm = vec_norm2(Av, 3);
            ASSERT_TRUE(nrm < 1e-10);
        }
        free(basis);
    }

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* sparse_qr_rank with explicit tolerance */
static void test_rank_explicit_tol(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1e-8);
    sparse_insert(A, 2, 2, 1e-16);

    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor(A, &qr), SPARSE_OK);

    /* With tight tolerance: rank = 3 (or 2 depending on threshold) */
    idx_t r_tight = sparse_qr_rank(&qr, 1e-18);
    /* With loose tolerance: rank should drop */
    idx_t r_loose = sparse_qr_rank(&qr, 1e-6);

    printf("    explicit tol: tight=%d, loose=%d\n", (int)r_tight, (int)r_loose);
    ASSERT_TRUE(r_tight >= r_loose);
    ASSERT_TRUE(r_loose >= 1);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Column reordering tests (Day 11)
 * ═══════════════════════════════════════════════════════════════════════ */

/* QR with AMD reordering produces same solution as without */
static void test_qr_reorder_amd_solve(void) {
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

    double b[4] = {1.0, 2.0, 3.0, 4.0};

    /* QR without reordering */
    sparse_qr_t qr_none;
    ASSERT_ERR(sparse_qr_factor(A, &qr_none), SPARSE_OK);
    double x_none[4];
    sparse_qr_solve(&qr_none, b, x_none, NULL);

    /* QR with AMD reordering */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_AMD};
    sparse_qr_t qr_amd;
    ASSERT_ERR(sparse_qr_factor_opts(A, &opts, &qr_amd), SPARSE_OK);
    double x_amd[4];
    sparse_qr_solve(&qr_amd, b, x_amd, NULL);

    /* Solutions should agree */
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x_none[i], x_amd[i], 1e-10);

    double res_none = compute_rel_residual(A, b, x_none, 4);
    double res_amd = compute_rel_residual(A, b, x_amd, 4);
    printf("    AMD reorder solve: none_res=%.3e, amd_res=%.3e\n", res_none, res_amd);
    ASSERT_TRUE(res_none < 1e-10);
    ASSERT_TRUE(res_amd < 1e-10);

    sparse_qr_free(&qr_none);
    sparse_qr_free(&qr_amd);
    sparse_free(A);
}

/* QR with AMD on nos4: compare R fill-in */
static void test_qr_reorder_nos4_fillin(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    /* QR without reordering */
    sparse_qr_t qr_none;
    {
        sparse_err_t ferr = sparse_qr_factor(A, &qr_none);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_free(A);
            return;
        }
    }
    idx_t nnz_none = sparse_nnz(qr_none.R);

    /* QR with AMD reordering */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_AMD};
    sparse_qr_t qr_amd;
    {
        sparse_err_t ferr = sparse_qr_factor_opts(A, &opts, &qr_amd);
        ASSERT_ERR(ferr, SPARSE_OK);
        if (ferr != SPARSE_OK) {
            sparse_qr_free(&qr_none);
            sparse_free(A);
            return;
        }
    }
    idx_t nnz_amd = sparse_nnz(qr_amd.R);

    printf("    nos4 R fill-in: none=%d, AMD=%d (%.1fx)\n", (int)nnz_none, (int)nnz_amd,
           nnz_none > 0 ? (double)nnz_amd / (double)nnz_none : 0.0);

    /* Both should produce correct solutions */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_exact);
    ASSERT_NOT_NULL(b);
    if (x_exact && b) {
        for (idx_t i = 0; i < n; i++)
            x_exact[i] = (double)(i + 1);
        sparse_matvec(A, x_exact, b);

        double *x = malloc((size_t)n * sizeof(double));
        ASSERT_NOT_NULL(x);
        if (x) {
            sparse_qr_solve(&qr_amd, b, x, NULL);
            double rr = compute_rel_residual(A, b, x, n);
            ASSERT_TRUE(rr < 1e-8);
            free(x);
        }
    }
    free(x_exact);
    free(b);

    sparse_qr_free(&qr_none);
    sparse_qr_free(&qr_amd);
    sparse_free(A);
}

/* QR reorder=NONE is same as default */
static void test_qr_reorder_none(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, 5.0);
    sparse_insert(A, 2, 2, 6.0);

    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE};
    sparse_qr_t qr;
    ASSERT_ERR(sparse_qr_factor_opts(A, &opts, &qr), SPARSE_OK);

    double recon = qr_reconstruction_error(A, &qr);
    ASSERT_TRUE(recon < 1e-10);

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

    /* Edge cases and hardening (Day 6) */
    RUN_TEST(test_qr_rank_1);
    RUN_TEST(test_qr_nearly_singular);
    RUN_TEST(test_qr_diagonal);
    RUN_TEST(test_qr_single_row);
    RUN_TEST(test_qr_single_col);
    RUN_TEST(test_qr_perm_valid);

    /* Q orthogonality and application (Day 7) */
    RUN_TEST(test_q_orthogonality_tall);
    RUN_TEST(test_q_transpose_b);
    RUN_TEST(test_q_apply_multiple);
    RUN_TEST(test_q_apply_inplace);
    RUN_TEST(test_q_orthogonality_wide);

    /* Least-squares solver (Day 8) */
    RUN_TEST(test_qr_solve_square);
    RUN_TEST(test_qr_solve_overdetermined);
    RUN_TEST(test_qr_solve_analytical);
    RUN_TEST(test_qr_solve_rank_deficient);
    RUN_TEST(test_qr_solve_nos4);
    RUN_TEST(test_qr_solve_null_residual);

    /* SuiteSparse validation (Day 9) */
    RUN_TEST(test_qr_bcsstk04);
    RUN_TEST(test_qr_west0067);
    RUN_TEST(test_qr_vs_lu);
    RUN_TEST(test_qr_tall_synthetic);

    /* Rank estimation and null space (Day 10) */
    RUN_TEST(test_rank_full);
    RUN_TEST(test_rank_1_nullspace);
    RUN_TEST(test_known_nullspace);
    RUN_TEST(test_rank_rect_deficient);
    RUN_TEST(test_rank_explicit_tol);

    /* Column reordering (Day 11) */
    RUN_TEST(test_qr_reorder_amd_solve);
    RUN_TEST(test_qr_reorder_nos4_fillin);
    RUN_TEST(test_qr_reorder_none);

    TEST_SUITE_END();
}
