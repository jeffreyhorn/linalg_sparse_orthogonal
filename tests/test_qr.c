/* _POSIX_C_SOURCE 200809L: needed for `popen` / `pclose` used by the
 * external dense-reference helper. Must be defined before any system header is
 * included so glibc's `<features.h>` sees it on first inclusion. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 200809L
#endif

#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"
#include "test_qr_helpers.h"
#define TF_ENABLE_EXTERNAL_REFERENCE_HELPER
#include "test_solver_helpers.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

static void assert_qr_reconstruction_below(const char *label, const SparseMatrix *A,
                                           const sparse_qr_t *qr, double tol);

static int read_qr_basis_external_reference(const char *fixture_key, double *values_out, idx_t n,
                                            char *reason, size_t reason_cap) {
    if (!reason || reason_cap == 0)
        return TF_EXTERNAL_REFERENCE_ERROR;
    if (!fixture_key || !values_out) {
        snprintf(reason, reason_cap, "external QR basis reference invalid arguments");
        return TF_EXTERNAL_REFERENCE_ERROR;
    }
    if (strcmp(fixture_key, "qr_economy_projector_5x3") != 0 &&
        strcmp(fixture_key, "qr_rank1_4x3_nullspace_projector") != 0 &&
        strcmp(fixture_key, "qr_rankdef_dependent_row_4x3_nullspace_projector") != 0 &&
        strcmp(fixture_key, "qr_rankdef_duplicate_5x4_nullspace_projector") != 0) {
        snprintf(reason, reason_cap, "unsupported external QR basis reference fixture key: %s",
                 fixture_key);
        return TF_EXTERNAL_REFERENCE_ERROR;
    }

    char cmd[256];
    int nw =
        snprintf(cmd, sizeof(cmd), "python3 tests/qr_external_dense_reference.py %s", fixture_key);
    if (nw < 0 || (size_t)nw >= sizeof(cmd)) {
        snprintf(reason, reason_cap, "external QR basis reference command overflow");
        return TF_EXTERNAL_REFERENCE_ERROR;
    }

    return (int)tf_read_external_reference_vector(cmd, "external QR basis reference", values_out, n,
                                                  reason, reason_cap);
}

static int read_qr_threshold_external_reference(const char *fixture_key, double *values_out,
                                                idx_t n, char *reason, size_t reason_cap) {
    if (!reason || reason_cap == 0)
        return TF_EXTERNAL_REFERENCE_ERROR;
    if (!fixture_key || !values_out) {
        snprintf(reason, reason_cap, "external QR threshold reference invalid arguments");
        return TF_EXTERNAL_REFERENCE_ERROR;
    }
    if (strcmp(fixture_key, "qr_rank_threshold_diag4_family") != 0 &&
        strcmp(fixture_key, "qr_rank_threshold_diag4_scaled_family") != 0 &&
        strcmp(fixture_key, "qr_rank_threshold_duplicate_5x4_perturbed_family") != 0) {
        snprintf(reason, reason_cap, "unsupported external QR threshold reference fixture key: %s",
                 fixture_key);
        return TF_EXTERNAL_REFERENCE_ERROR;
    }

    char cmd[256];
    int nw =
        snprintf(cmd, sizeof(cmd), "python3 tests/qr_external_dense_reference.py %s", fixture_key);
    if (nw < 0 || (size_t)nw >= sizeof(cmd)) {
        snprintf(reason, reason_cap, "external QR threshold reference command overflow");
        return TF_EXTERNAL_REFERENCE_ERROR;
    }

    return (int)tf_read_external_reference_vector(cmd, "external QR threshold reference",
                                                  values_out, n, reason, reason_cap);
}

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
    if (!qr.R) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
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
    assert_qr_reconstruction_below("3x3 reconstruction", A, &qr, 1e-10);
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

static void test_qr_public_scalar_alias(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 3.0);

    sparse_qr_t qr = {0};
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_scalar_t b[2] = {4.0, 9.0};
    sparse_scalar_t x[2] = {0.0, 0.0};
    sparse_scalar_t residual = -1.0;
    sparse_scalar_t diag[2] = {0.0, 0.0};

    ASSERT_EQ(sparse_scalar_bits(), sizeof(sparse_scalar_t) * CHAR_BIT);
    ASSERT_ERR(sparse_qr_solve(&qr, b, x, &residual), SPARSE_OK);
    ASSERT_ERR(sparse_qr_diag_r(&qr, diag), SPARSE_OK);

    ASSERT_NEAR(x[0], 2.0, 1e-12);
    ASSERT_NEAR(x[1], 3.0, 1e-12);
    ASSERT_NEAR(residual, 0.0, 1e-12);
    ASSERT_TRUE(fabs(diag[0]) > 0.0);
    ASSERT_TRUE(fabs(diag[1]) > 0.0);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_rejects_factored_matrix_reuse(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 2, 2.0);

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    if (!LU) {
        sparse_free(A);
        return;
    }
    REQUIRE_OK(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12));

    sparse_qr_t qr = {0};
    ASSERT_ERR(sparse_qr_factor(LU, &qr), SPARSE_ERR_BADARG);

    REQUIRE_OK(sparse_qr_factor(A, &qr));
    ASSERT_EQ(qr.rank, 3);

    sparse_qr_free(&qr);
    sparse_free(LU);
    sparse_free(A);
}

/* Q application: Q*Q^T*x = x */
static void test_q_roundtrip(void) {
    SparseMatrix *A = tf_qr_make_small_banded_4x3(0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

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
        if (qr.col_perm[i] >= 0 && qr.col_perm[i] < 4)
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
    assert_qr_reconstruction_below("3x5 reconstruction", A, &qr, 1e-10);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Rank-deficient: duplicate columns → rank < n */
static void test_qr_rank_deficient(void) {
    SparseMatrix *A = tf_qr_make_duplicate_column_4x3(2.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

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
    assert_qr_reconstruction_below("10x8 reconstruction", A, &qr, 1e-10);

    /* Verify R diagonal ordering */
    for (idx_t i = 0; i < qr.rank - 1; i++) {
        double ri = fabs(sparse_get_phys(qr.R, i, i));
        double ri1 = fabs(sparse_get_phys(qr.R, i + 1, i + 1));
        ASSERT_TRUE(ri >= ri1 - 1e-10);
    }

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Edge cases and hardening (Day 6)
 * ═══════════════════════════════════════════════════════════════════════ */

static void assert_qr_reconstruction_below(const char *label, const SparseMatrix *A,
                                           const sparse_qr_t *qr, double tol) {
    double recon_err = tf_qr_reconstruction_max_error(A, qr);
    printf("    %s: %.3e\n", label, recon_err);
    ASSERT_TRUE(recon_err < tol);
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

    assert_qr_reconstruction_below("rank-1 reconstruction", A, &qr, 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Nearly singular: one column is almost a multiple of another */
static void test_qr_nearly_singular(void) {
    SparseMatrix *A = tf_qr_make_near_duplicate_4x3(1e-12);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

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

    assert_qr_reconstruction_below("nearly singular reconstruction", A, &qr, 1e-8);

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

    assert_qr_reconstruction_below("diagonal reconstruction", A, &qr, 1e-10);

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
            idx_t p = qr.col_perm[i];
            ASSERT_TRUE(p >= 0 && p < qr.n);
            if (p >= 0 && p < qr.n)
                seen[p] = 1;
        }
        for (idx_t i = 0; i < qr.n; i++)
            ASSERT_TRUE(seen[i]);
        free(seen);
    }

    printf("    6x5 perm valid: rank=%d\n", (int)qr.rank);
    assert_qr_reconstruction_below("6x5 perm valid reconstruction", A, &qr, 1e-10);

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
    SparseMatrix *A = tf_qr_make_small_banded_4x3(1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

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
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

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
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

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
    SparseMatrix *A = tf_qr_make_duplicate_column_4x3(1.0);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

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

static void test_qr_external_dense_reference_rank1_4x3_nullspace_projector(void) {
#ifdef _WIN32
    SKIP_TEST("external QR dense reference helper is not enabled on Windows");
#else
    enum { QR_RANK1_NULLSPACE_N = 3 };
    enum { QR_RANK1_NULLSPACE_NULLITY = 2 };
    enum { QR_RANK1_NULLSPACE_PROJECTOR_VALUES = 4 + QR_RANK1_NULLSPACE_N * QR_RANK1_NULLSPACE_N };
    double ref[QR_RANK1_NULLSPACE_PROJECTOR_VALUES] = {0.0};
    char reason[256] = {0};
    int ref_status = read_qr_basis_external_reference("qr_rank1_4x3_nullspace_projector", ref,
                                                      QR_RANK1_NULLSPACE_PROJECTOR_VALUES, reason,
                                                      sizeof(reason));
    if (ref_status == TF_EXTERNAL_REFERENCE_SKIP)
        SKIP_TEST(reason);
    if (ref_status != TF_EXTERNAL_REFERENCE_OK) {
        TF_FAIL_("external QR basis reference failed: %s", reason);
        return;
    }

    ASSERT_EQ((idx_t)ref[0], QR_RANK1_NULLSPACE_N);
    ASSERT_EQ((idx_t)ref[1], 1);
    ASSERT_EQ((idx_t)ref[2], QR_RANK1_NULLSPACE_NULLITY);
    ASSERT_NEAR(ref[3], 0.0, 0.0);
    if ((idx_t)ref[0] != QR_RANK1_NULLSPACE_N || (idx_t)ref[1] != 1 ||
        (idx_t)ref[2] != QR_RANK1_NULLSPACE_NULLITY || ref[3] != 0.0)
        return;

    SparseMatrix *A = sparse_create(4, QR_RANK1_NULLSPACE_N);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < 4; i++) {
        for (idx_t j = 0; j < QR_RANK1_NULLSPACE_N; j++) {
            if (!tf_qr_insert_or_free(&A, i, j, (double)(i + 1)))
                return;
        }
    }

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 1);
    idx_t ndim = -1;
    sparse_err_t nullity_err = sparse_qr_nullspace(&qr, ref[3], NULL, &ndim);
    ASSERT_ERR(nullity_err, SPARSE_OK);
    if (nullity_err != SPARSE_OK) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    ASSERT_EQ(ndim, QR_RANK1_NULLSPACE_NULLITY);
    if (qr.rank != 1 || ndim != QR_RANK1_NULLSPACE_NULLITY) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double basis[QR_RANK1_NULLSPACE_N * QR_RANK1_NULLSPACE_NULLITY] = {0.0};
    sparse_err_t basis_err = sparse_qr_nullspace(&qr, ref[3], basis, &ndim);
    ASSERT_ERR(basis_err, SPARSE_OK);
    if (basis_err != SPARSE_OK) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    if (ndim != QR_RANK1_NULLSPACE_NULLITY) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double z[QR_RANK1_NULLSPACE_N * QR_RANK1_NULLSPACE_NULLITY] = {0.0};
    for (idx_t i = 0; i < QR_RANK1_NULLSPACE_N; i++)
        z[i] = basis[i];
    double norm0 = vec_norm2(z, QR_RANK1_NULLSPACE_N);
    ASSERT_TRUE(norm0 > 0.0);
    if (norm0 <= 0.0) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < QR_RANK1_NULLSPACE_N; i++)
        z[i] /= norm0;

    double *z1 = &z[QR_RANK1_NULLSPACE_N];
    const double *basis1 = &basis[QR_RANK1_NULLSPACE_N];
    double dot01 = 0.0;
    for (idx_t i = 0; i < QR_RANK1_NULLSPACE_N; i++)
        dot01 += basis1[i] * z[i];
    for (idx_t i = 0; i < QR_RANK1_NULLSPACE_N; i++)
        z1[i] = basis1[i] - dot01 * z[i];
    double norm1 = vec_norm2(z1, QR_RANK1_NULLSPACE_N);
    ASSERT_TRUE(norm1 > 0.0);
    if (norm1 <= 0.0) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < QR_RANK1_NULLSPACE_N; i++)
        z1[i] /= norm1;

    double max_null_residual = 0.0;
    for (idx_t col = 0; col < QR_RANK1_NULLSPACE_NULLITY; col++) {
        double Av[4] = {0.0};
        sparse_matvec(A, &z[(size_t)col * QR_RANK1_NULLSPACE_N], Av);
        double residual = vec_norm2(Av, 4);
        if (residual > max_null_residual)
            max_null_residual = residual;
    }

    double max_orthogonality_err = 0.0;
    for (idx_t col_i = 0; col_i < QR_RANK1_NULLSPACE_NULLITY; col_i++) {
        for (idx_t col_j = 0; col_j < QR_RANK1_NULLSPACE_NULLITY; col_j++) {
            double dot = 0.0;
            for (idx_t row = 0; row < QR_RANK1_NULLSPACE_N; row++) {
                dot += z[(size_t)col_i * QR_RANK1_NULLSPACE_N + (size_t)row] *
                       z[(size_t)col_j * QR_RANK1_NULLSPACE_N + (size_t)row];
            }
            double expected = col_i == col_j ? 1.0 : 0.0;
            double diff = fabs(dot - expected);
            if (diff > max_orthogonality_err)
                max_orthogonality_err = diff;
        }
    }

    double max_projector_diff = 0.0;
    for (idx_t row = 0; row < QR_RANK1_NULLSPACE_N; row++) {
        for (idx_t col = 0; col < QR_RANK1_NULLSPACE_N; col++) {
            double product = 0.0;
            for (idx_t k = 0; k < QR_RANK1_NULLSPACE_NULLITY; k++)
                product += z[(size_t)k * QR_RANK1_NULLSPACE_N + (size_t)row] *
                           z[(size_t)k * QR_RANK1_NULLSPACE_N + (size_t)col];
            double diff = fabs(product - ref[4 + (size_t)col * QR_RANK1_NULLSPACE_N + (size_t)row]);
            if (diff > max_projector_diff)
                max_projector_diff = diff;
        }
    }

    printf("    external QR dense ref rank1_4x3_nullspace_projector: "
           "projector diff = %.3e, null residual = %.3e, orthogonality err = %.3e\n",
           max_projector_diff, max_null_residual, max_orthogonality_err);
    ASSERT_TRUE(max_projector_diff < 1e-8);
    ASSERT_TRUE(max_null_residual < 1e-10);
    ASSERT_TRUE(max_orthogonality_err < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
#endif
}

static void test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector(void) {
#ifdef _WIN32
    SKIP_TEST("external QR dense reference helper is not enabled on Windows");
#else
    enum { QR_NULLSPACE_N = 4 };
    enum { QR_NULLSPACE_PROJECTOR_VALUES = 4 + QR_NULLSPACE_N * QR_NULLSPACE_N };
    double ref[QR_NULLSPACE_PROJECTOR_VALUES] = {0.0};
    char reason[256] = {0};
    int ref_status =
        read_qr_basis_external_reference("qr_rankdef_duplicate_5x4_nullspace_projector", ref,
                                         QR_NULLSPACE_PROJECTOR_VALUES, reason, sizeof(reason));
    if (ref_status == TF_EXTERNAL_REFERENCE_SKIP)
        SKIP_TEST(reason);
    if (ref_status != TF_EXTERNAL_REFERENCE_OK) {
        TF_FAIL_("external QR basis reference failed: %s", reason);
        return;
    }

    ASSERT_EQ((idx_t)ref[0], QR_NULLSPACE_N);
    ASSERT_EQ((idx_t)ref[1], 3);
    ASSERT_EQ((idx_t)ref[2], 1);
    ASSERT_NEAR(ref[3], 0.0, 0.0);
    if ((idx_t)ref[0] != QR_NULLSPACE_N || (idx_t)ref[1] != 3 || (idx_t)ref[2] != 1 ||
        ref[3] != 0.0)
        return;

    SparseMatrix *A = tf_qr_make_rankdef_duplicate_5x4();
    if (!A)
        return;

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 3);
    idx_t ndim = -1;
    ASSERT_ERR(sparse_qr_nullspace(&qr, ref[3], NULL, &ndim), SPARSE_OK);
    ASSERT_EQ(ndim, 1);
    if (qr.rank != 3 || ndim != 1) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double basis[QR_NULLSPACE_N] = {0.0};
    ASSERT_ERR(sparse_qr_nullspace(&qr, ref[3], basis, &ndim), SPARSE_OK);
    if (ndim != 1) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double norm_sq = 0.0;
    for (idx_t i = 0; i < QR_NULLSPACE_N; i++)
        norm_sq += basis[i] * basis[i];
    ASSERT_TRUE(norm_sq > 0.0);
    if (norm_sq <= 0.0) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double basis_norm = sqrt(norm_sq);
    double unit_basis[QR_NULLSPACE_N] = {0.0};
    for (idx_t i = 0; i < QR_NULLSPACE_N; i++)
        unit_basis[i] = basis[i] / basis_norm;

    double Av[5] = {0.0};
    sparse_matvec(A, unit_basis, Av);
    double null_residual = vec_norm2(Av, 5);
    double max_projector_diff = 0.0;
    for (idx_t i = 0; i < QR_NULLSPACE_N; i++) {
        for (idx_t j = 0; j < QR_NULLSPACE_N; j++) {
            double product = unit_basis[i] * unit_basis[j];
            double diff = fabs(product - ref[4 + (size_t)j * QR_NULLSPACE_N + (size_t)i]);
            if (diff > max_projector_diff)
                max_projector_diff = diff;
        }
    }
    double unit_norm_sq = 0.0;
    for (idx_t i = 0; i < QR_NULLSPACE_N; i++)
        unit_norm_sq += unit_basis[i] * unit_basis[i];
    double orthogonality_err = fabs(unit_norm_sq - 1.0);

    printf("    external QR dense ref rankdef_duplicate_5x4_nullspace_projector: "
           "projector diff = %.3e, null residual = %.3e, norm err = %.3e\n",
           max_projector_diff, null_residual, orthogonality_err);
    ASSERT_TRUE(max_projector_diff < 1e-8);
    ASSERT_TRUE(null_residual < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
#endif
}

static void test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector(void) {
#ifdef _WIN32
    SKIP_TEST("external QR dense reference helper is not enabled on Windows");
#else
    enum { QR_DEPENDENT_ROW_NULLSPACE_N = 3 };
    enum {
        QR_DEPENDENT_ROW_NULLSPACE_PROJECTOR_VALUES =
            4 + QR_DEPENDENT_ROW_NULLSPACE_N * QR_DEPENDENT_ROW_NULLSPACE_N
    };
    double ref[QR_DEPENDENT_ROW_NULLSPACE_PROJECTOR_VALUES] = {0.0};
    char reason[256] = {0};
    int ref_status = read_qr_basis_external_reference(
        "qr_rankdef_dependent_row_4x3_nullspace_projector", ref,
        QR_DEPENDENT_ROW_NULLSPACE_PROJECTOR_VALUES, reason, sizeof(reason));
    if (ref_status == TF_EXTERNAL_REFERENCE_SKIP)
        SKIP_TEST(reason);
    if (ref_status != TF_EXTERNAL_REFERENCE_OK) {
        TF_FAIL_("external QR basis reference failed: %s", reason);
        return;
    }

    ASSERT_EQ((idx_t)ref[0], QR_DEPENDENT_ROW_NULLSPACE_N);
    ASSERT_EQ((idx_t)ref[1], 2);
    ASSERT_EQ((idx_t)ref[2], 1);
    ASSERT_NEAR(ref[3], 0.0, 0.0);
    if ((idx_t)ref[0] != QR_DEPENDENT_ROW_NULLSPACE_N || (idx_t)ref[1] != 2 || (idx_t)ref[2] != 1 ||
        ref[3] != 0.0)
        return;

    SparseMatrix *A = tf_qr_make_dependent_row_4x3();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 2);
    idx_t ndim = -1;
    sparse_err_t nullity_err = sparse_qr_nullspace(&qr, ref[3], NULL, &ndim);
    ASSERT_ERR(nullity_err, SPARSE_OK);
    if (nullity_err != SPARSE_OK) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    ASSERT_EQ(ndim, 1);
    if (qr.rank != 2 || ndim != 1) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double basis[QR_DEPENDENT_ROW_NULLSPACE_N] = {0.0};
    sparse_err_t basis_err = sparse_qr_nullspace(&qr, ref[3], basis, &ndim);
    ASSERT_ERR(basis_err, SPARSE_OK);
    if (basis_err != SPARSE_OK) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    if (ndim != 1) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double norm_sq = 0.0;
    for (idx_t i = 0; i < QR_DEPENDENT_ROW_NULLSPACE_N; i++)
        norm_sq += basis[i] * basis[i];
    ASSERT_TRUE(norm_sq > 0.0);
    if (norm_sq <= 0.0) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double basis_norm = sqrt(norm_sq);
    double unit_basis[QR_DEPENDENT_ROW_NULLSPACE_N] = {0.0};
    for (idx_t i = 0; i < QR_DEPENDENT_ROW_NULLSPACE_N; i++)
        unit_basis[i] = basis[i] / basis_norm;

    double Av[4] = {0.0};
    sparse_matvec(A, unit_basis, Av);
    double null_residual = vec_norm2(Av, 4);

    double max_projector_diff = 0.0;
    for (idx_t row = 0; row < QR_DEPENDENT_ROW_NULLSPACE_N; row++) {
        for (idx_t col = 0; col < QR_DEPENDENT_ROW_NULLSPACE_N; col++) {
            double product = unit_basis[row] * unit_basis[col];
            double diff =
                fabs(product - ref[4 + (size_t)col * QR_DEPENDENT_ROW_NULLSPACE_N + (size_t)row]);
            if (diff > max_projector_diff)
                max_projector_diff = diff;
        }
    }

    double unit_norm_sq = 0.0;
    for (idx_t i = 0; i < QR_DEPENDENT_ROW_NULLSPACE_N; i++)
        unit_norm_sq += unit_basis[i] * unit_basis[i];
    double orthogonality_err = fabs(unit_norm_sq - 1.0);

    printf("    external QR dense ref rankdef_dependent_row_4x3_nullspace_projector: "
           "projector diff = %.3e, null residual = %.3e, norm err = %.3e\n",
           max_projector_diff, null_residual, orthogonality_err);
    ASSERT_TRUE(max_projector_diff < 1e-8);
    ASSERT_TRUE(null_residual < 1e-10);
    ASSERT_TRUE(orthogonality_err < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
#endif
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
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

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
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

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

static void test_qr_rank_dependent_row_fixture(void) {
    SparseMatrix *A = tf_qr_make_dependent_row_4x3();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    ASSERT_EQ(qr.rank, 2);
    ASSERT_EQ(sparse_qr_rank(&qr, 0.0), 2);
    assert_qr_reconstruction_below("dependent-row-rank-def", A, &qr, 1e-10);

    idx_t ndim = -1;
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, NULL, &ndim), SPARSE_OK);
    ASSERT_EQ(ndim, 1);
    if (ndim == 1) {
        double basis[3];
        ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, basis, &ndim), SPARSE_OK);
        double Av[4];
        sparse_matvec(A, basis, Av);
        double nrm = vec_norm2(Av, 4);
        ASSERT_TRUE(nrm < 1e-10);
        printf("    dependent-row rank fixture: rank=%d, null residual=%.3e\n", (int)qr.rank, nrm);
    }

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_rank_diagonal_threshold_fixture(void) {
    const double diag[4] = {1.0, 1e-8, 1e-12, 0.0};
    SparseMatrix *A = tf_qr_make_diag_matrix(4, 4, diag, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    idx_t rank_1e14 = sparse_qr_rank(&qr, 1e-14);
    idx_t rank_1e10 = sparse_qr_rank(&qr, 1e-10);
    idx_t rank_1e6 = sparse_qr_rank(&qr, 1e-6);
    ASSERT_EQ(rank_1e14, 3);
    ASSERT_EQ(rank_1e10, 2);
    ASSERT_EQ(rank_1e6, 1);
    printf("    QR diag threshold fixture: rank(1e-14)=%d, rank(1e-10)=%d, rank(1e-6)=%d\n",
           (int)rank_1e14, (int)rank_1e10, (int)rank_1e6);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_external_dense_reference_rank_threshold_diag4_family(void) {
#ifdef _WIN32
    SKIP_TEST("external QR dense reference helper is not enabled on Windows");
#else
    enum { QR_THRESHOLD_VALUES = 6 };
    const double diag[4] = {1.0, 1e-8, 1e-12, 0.0};
    double ref[QR_THRESHOLD_VALUES] = {0.0};
    char reason[256] = {0};
    int ref_status = read_qr_threshold_external_reference(
        "qr_rank_threshold_diag4_family", ref, QR_THRESHOLD_VALUES, reason, sizeof(reason));
    if (ref_status == TF_EXTERNAL_REFERENCE_SKIP)
        SKIP_TEST(reason);
    if (ref_status != TF_EXTERNAL_REFERENCE_OK) {
        TF_FAIL_("external QR threshold reference failed: %s", reason);
        return;
    }

    SparseMatrix *A = tf_qr_make_diag_matrix(4, 4, diag, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double rdiag[4] = {0.0};
    ASSERT_ERR(sparse_qr_diag_r(&qr, rdiag), SPARSE_OK);
    double r00 = fabs(rdiag[0]);
    const double expected_thresholds[QR_THRESHOLD_VALUES / 2] = {1e-14, 1e-10, 1e-6};

    for (idx_t i = 0; i < QR_THRESHOLD_VALUES / 2; i++) {
        double threshold = ref[(size_t)i * 2];
        idx_t expected_rank = (idx_t)ref[(size_t)i * 2 + 1];
        ASSERT_NEAR(threshold, expected_thresholds[i], 0.0);
        if (threshold != expected_thresholds[i]) {
            sparse_qr_free(&qr);
            sparse_free(A);
            return;
        }
        idx_t product_rank = sparse_qr_rank(&qr, threshold);
        sparse_qr_rank_info_t info;
        sparse_err_t info_err = sparse_qr_rank_info(&qr, threshold, &info);
        ASSERT_ERR(info_err, SPARSE_OK);
        if (info_err != SPARSE_OK) {
            sparse_qr_free(&qr);
            sparse_free(A);
            return;
        }
        double abs_threshold = threshold * r00;
        printf("    external QR dense ref rank_threshold_diag4_family: "
               "tol=%.0e abs_tol=%.3e expected=%d product=%d info=%d "
               "|Rdiag|=[%.3e, %.3e, %.3e, %.3e]\n",
               threshold, abs_threshold, (int)expected_rank, (int)product_rank, (int)info.rank,
               fabs(rdiag[0]), fabs(rdiag[1]), fabs(rdiag[2]), fabs(rdiag[3]));
        ASSERT_EQ(product_rank, expected_rank);
        ASSERT_EQ(info.rank, expected_rank);
    }

    sparse_qr_free(&qr);
    sparse_free(A);
#endif
}

static void test_qr_external_dense_reference_rank_threshold_diag4_scaled_family(void) {
#ifdef _WIN32
    SKIP_TEST("external QR dense reference helper is not enabled on Windows");
#else
    enum { QR_THRESHOLD_SCALED_RECORDS = 9 };
    enum { QR_THRESHOLD_SCALED_FIELDS = 3 };
    enum { QR_THRESHOLD_SCALED_VALUES = QR_THRESHOLD_SCALED_RECORDS * QR_THRESHOLD_SCALED_FIELDS };
    double ref[QR_THRESHOLD_SCALED_VALUES] = {0.0};
    char reason[256] = {0};
    int ref_status =
        read_qr_threshold_external_reference("qr_rank_threshold_diag4_scaled_family", ref,
                                             QR_THRESHOLD_SCALED_VALUES, reason, sizeof(reason));
    if (ref_status == TF_EXTERNAL_REFERENCE_SKIP)
        SKIP_TEST(reason);
    if (ref_status != TF_EXTERNAL_REFERENCE_OK) {
        TF_FAIL_("external QR threshold reference failed: %s", reason);
        return;
    }

    const double expected_scales[3] = {1e-6, 1.0, 1e6};
    const double expected_thresholds[3] = {1e-14, 1e-10, 1e-6};
    const idx_t expected_ranks[3] = {3, 2, 1};

    for (idx_t scale_idx = 0; scale_idx < 3; scale_idx++) {
        double scale = expected_scales[scale_idx];
        const double diag[4] = {scale, scale * 1e-8, scale * 1e-12, 0.0};
        SparseMatrix *A = tf_qr_make_diag_matrix(4, 4, diag, 4);
        ASSERT_NOT_NULL(A);
        if (!A)
            return;

        sparse_qr_t qr;
        sparse_err_t err = sparse_qr_factor(A, &qr);
        ASSERT_ERR(err, SPARSE_OK);
        if (err != SPARSE_OK) {
            sparse_free(A);
            return;
        }

        double rdiag[4] = {0.0};
        sparse_err_t diag_err = sparse_qr_diag_r(&qr, rdiag);
        ASSERT_ERR(diag_err, SPARSE_OK);
        if (diag_err != SPARSE_OK) {
            sparse_qr_free(&qr);
            sparse_free(A);
            return;
        }
        double r00 = fabs(rdiag[0]);

        for (idx_t threshold_idx = 0; threshold_idx < 3; threshold_idx++) {
            size_t record = (size_t)scale_idx * 3u + (size_t)threshold_idx;
            double ref_scale = ref[record * QR_THRESHOLD_SCALED_FIELDS];
            double threshold = ref[record * QR_THRESHOLD_SCALED_FIELDS + 1u];
            idx_t expected_rank = (idx_t)ref[record * QR_THRESHOLD_SCALED_FIELDS + 2u];
            ASSERT_NEAR(ref_scale, expected_scales[scale_idx], 0.0);
            ASSERT_NEAR(threshold, expected_thresholds[threshold_idx], 0.0);
            ASSERT_EQ(expected_rank, expected_ranks[threshold_idx]);
            if (ref_scale != expected_scales[scale_idx] ||
                threshold != expected_thresholds[threshold_idx] ||
                expected_rank != expected_ranks[threshold_idx]) {
                sparse_qr_free(&qr);
                sparse_free(A);
                return;
            }

            idx_t product_rank = sparse_qr_rank(&qr, threshold);
            sparse_qr_rank_info_t info;
            sparse_err_t info_err = sparse_qr_rank_info(&qr, threshold, &info);
            ASSERT_ERR(info_err, SPARSE_OK);
            if (info_err != SPARSE_OK) {
                sparse_qr_free(&qr);
                sparse_free(A);
                return;
            }
            double abs_threshold = threshold * r00;
            printf("    external QR dense ref rank_threshold_diag4_scaled_family: "
                   "scale=%.0e tol=%.0e abs_tol=%.3e expected=%d product=%d info=%d "
                   "|Rdiag|=[%.3e, %.3e, %.3e, %.3e]\n",
                   scale, threshold, abs_threshold, (int)expected_rank, (int)product_rank,
                   (int)info.rank, fabs(rdiag[0]), fabs(rdiag[1]), fabs(rdiag[2]), fabs(rdiag[3]));
            ASSERT_EQ(product_rank, expected_rank);
            ASSERT_EQ(info.rank, expected_rank);
        }

        sparse_qr_free(&qr);
        sparse_free(A);
    }
#endif
}

static void test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family(void) {
#ifdef _WIN32
    SKIP_TEST("external QR dense reference helper is not enabled on Windows");
#else
    enum { QR_THRESHOLD_PERTURBED_RECORDS = 2 };
    enum { QR_THRESHOLD_PERTURBED_FIELDS = 3 };
    enum {
        QR_THRESHOLD_PERTURBED_VALUES =
            QR_THRESHOLD_PERTURBED_RECORDS * QR_THRESHOLD_PERTURBED_FIELDS
    };
    double ref[QR_THRESHOLD_PERTURBED_VALUES] = {0.0};
    char reason[256] = {0};
    int ref_status = read_qr_threshold_external_reference(
        "qr_rank_threshold_duplicate_5x4_perturbed_family", ref, QR_THRESHOLD_PERTURBED_VALUES,
        reason, sizeof(reason));
    if (ref_status == TF_EXTERNAL_REFERENCE_SKIP)
        SKIP_TEST(reason);
    if (ref_status != TF_EXTERNAL_REFERENCE_OK) {
        TF_FAIL_("external QR threshold reference failed: %s", reason);
        return;
    }

    const double expected_perturbation = 6e-8;
    const double expected_thresholds[QR_THRESHOLD_PERTURBED_RECORDS] = {1e-10, 1e-6};
    const idx_t expected_ranks[QR_THRESHOLD_PERTURBED_RECORDS] = {4, 3};

    SparseMatrix *A = tf_qr_make_rankdef_duplicate_5x4();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_err_t insert_err = sparse_insert(A, 0, 3, expected_perturbation);
    ASSERT_ERR(insert_err, SPARSE_OK);
    if (insert_err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double rdiag[4] = {0.0};
    sparse_err_t diag_err = sparse_qr_diag_r(&qr, rdiag);
    ASSERT_ERR(diag_err, SPARSE_OK);
    if (diag_err != SPARSE_OK) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    double r00 = fabs(rdiag[0]);
    double pivot_ratio = r00 > 0.0 ? fabs(rdiag[3]) / r00 : 0.0;

    for (idx_t i = 0; i < QR_THRESHOLD_PERTURBED_RECORDS; i++) {
        double perturbation = ref[(size_t)i * QR_THRESHOLD_PERTURBED_FIELDS];
        double threshold = ref[(size_t)i * QR_THRESHOLD_PERTURBED_FIELDS + 1u];
        idx_t expected_rank = (idx_t)ref[(size_t)i * QR_THRESHOLD_PERTURBED_FIELDS + 2u];
        ASSERT_NEAR(perturbation, expected_perturbation, 0.0);
        ASSERT_NEAR(threshold, expected_thresholds[i], 0.0);
        ASSERT_EQ(expected_rank, expected_ranks[i]);
        if (perturbation != expected_perturbation || threshold != expected_thresholds[i] ||
            expected_rank != expected_ranks[i]) {
            sparse_qr_free(&qr);
            sparse_free(A);
            return;
        }

        idx_t product_rank = sparse_qr_rank(&qr, threshold);
        sparse_qr_rank_info_t info;
        sparse_err_t info_err = sparse_qr_rank_info(&qr, threshold, &info);
        ASSERT_ERR(info_err, SPARSE_OK);
        if (info_err != SPARSE_OK) {
            sparse_qr_free(&qr);
            sparse_free(A);
            return;
        }
        double abs_threshold = threshold * r00;
        printf("    external QR dense ref rank_threshold_duplicate_5x4_perturbed_family: "
               "perturb=%.1e tol=%.0e abs_tol=%.3e expected=%d product=%d info=%d "
               "pivot_ratio=%.3e |Rdiag|=[%.3e, %.3e, %.3e, %.3e]\n",
               perturbation, threshold, abs_threshold, (int)expected_rank, (int)product_rank,
               (int)info.rank, pivot_ratio, fabs(rdiag[0]), fabs(rdiag[1]), fabs(rdiag[2]),
               fabs(rdiag[3]));
        ASSERT_EQ(product_rank, expected_rank);
        ASSERT_EQ(info.rank, expected_rank);
    }

    sparse_qr_free(&qr);
    sparse_free(A);
#endif
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
    sparse_err_t err_none = sparse_qr_factor(A, &qr_none);
    ASSERT_ERR(err_none, SPARSE_OK);
    if (err_none != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    double x_none[4];
    sparse_qr_solve(&qr_none, b, x_none, NULL);

    /* QR with AMD reordering */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_AMD};
    sparse_qr_t qr_amd;
    sparse_err_t err_amd = sparse_qr_factor_opts(A, &opts, &qr_amd);
    ASSERT_ERR(err_amd, SPARSE_OK);
    if (err_amd != SPARSE_OK) {
        sparse_qr_free(&qr_none);
        sparse_free(A);
        return;
    }
    double x_amd[4];
    sparse_qr_solve(&qr_amd, b, x_amd, NULL);

    /* Solutions should agree */
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x_none[i], x_amd[i], 1e-10);

    double res_none = tf_qr_relative_residual_l2(A, b, x_none, 4);
    double res_amd = tf_qr_relative_residual_l2(A, b, x_amd, 4);
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
    double *x_exact = NULL;
    double *b = NULL;
    if (tf_qr_make_exact_rhs(A, n, n, &x_exact, &b)) {
        double *x = malloc((size_t)n * sizeof(double));
        ASSERT_NOT_NULL(x);
        if (x) {
            sparse_qr_solve(&qr_amd, b, x, NULL);
            double rr = tf_qr_relative_residual_l2(A, b, x, n);
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
    sparse_err_t err = sparse_qr_factor_opts(A, &opts, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double recon = tf_qr_reconstruction_max_error(A, &qr);
    ASSERT_TRUE(recon < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Economy QR tests (Sprint 7 Day 7)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Economy QR solve matches full QR solve on tall-skinny matrix */
static void test_economy_solve_tall(void) {
    idx_t m = 50, nc = 10;
    SparseMatrix *A = tf_qr_make_tall_diagonal_dominant(m, nc, 10.0, 1.0, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    double *b = malloc((size_t)m * sizeof(double));
    ASSERT_NOT_NULL(b);
    if (!b) {
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < m; i++)
        b[i] = (double)(i + 1);

    /* Full QR */
    sparse_qr_t qr_full;
    sparse_err_t err = sparse_qr_factor(A, &qr_full);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(b);
        sparse_free(A);
        return;
    }
    double *x_full = malloc((size_t)nc * sizeof(double));
    ASSERT_NOT_NULL(x_full);
    if (!x_full) {
        free(b);
        sparse_qr_free(&qr_full);
        sparse_free(A);
        return;
    }
    double res_full = 0.0;
    sparse_qr_solve(&qr_full, b, x_full, &res_full);

    /* Economy QR */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 1};
    sparse_qr_t qr_econ;
    err = sparse_qr_factor_opts(A, &opts, &qr_econ);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(x_full);
        free(b);
        sparse_qr_free(&qr_full);
        sparse_free(A);
        return;
    }
    double *x_econ = malloc((size_t)nc * sizeof(double));
    ASSERT_NOT_NULL(x_econ);
    if (!x_econ) {
        free(x_full);
        free(b);
        sparse_qr_free(&qr_full);
        sparse_qr_free(&qr_econ);
        sparse_free(A);
        return;
    }
    double res_econ = 0.0;
    sparse_qr_solve(&qr_econ, b, x_econ, &res_econ);

    ASSERT_TRUE(qr_econ.economy);

    {
        printf("    economy solve 50x10: full_res=%.3e, econ_res=%.3e\n", res_full, res_econ);
        for (idx_t i = 0; i < nc; i++)
            ASSERT_NEAR(x_full[i], x_econ[i], 1e-10);
        ASSERT_NEAR(res_full, res_econ, 1e-10);
    }

    free(x_full);
    free(x_econ);
    free(b);
    sparse_qr_free(&qr_full);
    sparse_qr_free(&qr_econ);
    sparse_free(A);
}

/* Economy Q orthogonality: thin Q^T * Q = I_n */
static void test_economy_q_orthogonality(void) {
    idx_t m = 20, nc = 5;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if (i == j || (i + j) % 3 == 0)
                sparse_insert(A, i, j, (double)(i + j + 1));

    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 1};
    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor_opts(A, &opts, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Form thin Q (m × nc) */
    double *Q = calloc((size_t)m * (size_t)nc, sizeof(double));
    ASSERT_NOT_NULL(Q);
    if (!Q) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_qr_form_q(&qr, Q);

    /* Check Q^T * Q ≈ I_nc */
    double max_err = 0.0;
    for (idx_t i = 0; i < nc; i++) {
        for (idx_t j = 0; j < nc; j++) {
            double dot = 0.0;
            for (idx_t k = 0; k < m; k++)
                dot += Q[(size_t)i * (size_t)m + (size_t)k] * Q[(size_t)j * (size_t)m + (size_t)k];
            double expected = (i == j) ? 1.0 : 0.0;
            double e = fabs(dot - expected);
            if (e > max_err)
                max_err = e;
        }
    }
    printf("    economy Q^T*Q orthogonality: max_err=%.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-10);

    free(Q);
    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_external_dense_reference_economy_projector_5x3(void) {
#ifdef _WIN32
    SKIP_TEST("external QR dense reference helper is not enabled on Windows");
#else
    enum { QR_ECONOMY_PROJECTOR_ROWS = 5, QR_ECONOMY_PROJECTOR_COLS = 3 };
    enum {
        QR_ECONOMY_PROJECTOR_VALUES = 4 + QR_ECONOMY_PROJECTOR_ROWS * QR_ECONOMY_PROJECTOR_ROWS
    };
    double ref[QR_ECONOMY_PROJECTOR_VALUES] = {0.0};
    char reason[256] = {0};
    int ref_status = read_qr_basis_external_reference(
        "qr_economy_projector_5x3", ref, QR_ECONOMY_PROJECTOR_VALUES, reason, sizeof(reason));
    if (ref_status == TF_EXTERNAL_REFERENCE_SKIP)
        SKIP_TEST(reason);
    if (ref_status != TF_EXTERNAL_REFERENCE_OK) {
        TF_FAIL_("external QR basis reference failed: %s", reason);
        return;
    }

    ASSERT_EQ((idx_t)ref[0], QR_ECONOMY_PROJECTOR_ROWS);
    ASSERT_EQ((idx_t)ref[1], QR_ECONOMY_PROJECTOR_COLS);
    ASSERT_EQ((idx_t)ref[2], QR_ECONOMY_PROJECTOR_COLS);
    ASSERT_EQ((idx_t)ref[3], QR_ECONOMY_PROJECTOR_COLS);

    SparseMatrix *A = sparse_create(QR_ECONOMY_PROJECTOR_ROWS, QR_ECONOMY_PROJECTOR_COLS);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 0, 2, 2.0) ||
        !tf_qr_insert_or_free(&A, 1, 1, 1.0) || !tf_qr_insert_or_free(&A, 1, 2, -1.0) ||
        !tf_qr_insert_or_free(&A, 2, 0, 2.0) || !tf_qr_insert_or_free(&A, 2, 1, -1.0) ||
        !tf_qr_insert_or_free(&A, 3, 0, 1.0) || !tf_qr_insert_or_free(&A, 3, 1, 1.0) ||
        !tf_qr_insert_or_free(&A, 3, 2, 1.0) || !tf_qr_insert_or_free(&A, 4, 0, 3.0) ||
        !tf_qr_insert_or_free(&A, 4, 2, -2.0))
        return;

    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 1};
    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor_opts(A, &opts, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    ASSERT_TRUE(qr.economy);
    ASSERT_NOT_NULL(qr.R);
    if (qr.R) {
        ASSERT_EQ(sparse_rows(qr.R), QR_ECONOMY_PROJECTOR_COLS);
        ASSERT_EQ(sparse_cols(qr.R), QR_ECONOMY_PROJECTOR_COLS);
    }

    double *Q = calloc((size_t)QR_ECONOMY_PROJECTOR_ROWS * (size_t)QR_ECONOMY_PROJECTOR_COLS,
                       sizeof(double));
    ASSERT_NOT_NULL(Q);
    if (!Q) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    err = sparse_qr_form_q(&qr, Q);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(Q);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double max_projector_diff = 0.0;
    double max_orthogonality_err = 0.0;
    for (idx_t i = 0; i < QR_ECONOMY_PROJECTOR_ROWS; i++) {
        for (idx_t j = 0; j < QR_ECONOMY_PROJECTOR_ROWS; j++) {
            double product = 0.0;
            for (idx_t col = 0; col < QR_ECONOMY_PROJECTOR_COLS; col++)
                product += Q[(size_t)col * QR_ECONOMY_PROJECTOR_ROWS + (size_t)i] *
                           Q[(size_t)col * QR_ECONOMY_PROJECTOR_ROWS + (size_t)j];
            double diff =
                fabs(product - ref[4 + (size_t)j * QR_ECONOMY_PROJECTOR_ROWS + (size_t)i]);
            if (diff > max_projector_diff)
                max_projector_diff = diff;
        }
    }

    for (idx_t i = 0; i < QR_ECONOMY_PROJECTOR_COLS; i++) {
        for (idx_t j = 0; j < QR_ECONOMY_PROJECTOR_COLS; j++) {
            double dot = 0.0;
            for (idx_t row = 0; row < QR_ECONOMY_PROJECTOR_ROWS; row++)
                dot += Q[(size_t)i * QR_ECONOMY_PROJECTOR_ROWS + (size_t)row] *
                       Q[(size_t)j * QR_ECONOMY_PROJECTOR_ROWS + (size_t)row];
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = fabs(dot - expected);
            if (diff > max_orthogonality_err)
                max_orthogonality_err = diff;
        }
    }

    printf("    external QR dense ref economy_projector_5x3: projector diff = %.3e, "
           "orthogonality err = %.3e\n",
           max_projector_diff, max_orthogonality_err);
    ASSERT_TRUE(max_projector_diff < 1e-8);
    ASSERT_TRUE(max_orthogonality_err < 1e-10);

    free(Q);
    sparse_qr_free(&qr);
    sparse_free(A);
#endif
}

/* Square matrix with economy=1: same as economy=0 */
static void test_economy_square(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 5.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.0);
    }

    double b[5] = {4.0, 3.0, 3.0, 3.0, 4.0};

    /* Full QR */
    sparse_qr_t qr_full;
    sparse_err_t err = sparse_qr_factor(A, &qr_full);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    double x_full[5];
    sparse_qr_solve(&qr_full, b, x_full, NULL);

    /* Economy QR */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 1};
    sparse_qr_t qr_econ;
    err = sparse_qr_factor_opts(A, &opts, &qr_econ);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_qr_free(&qr_full);
        sparse_free(A);
        return;
    }
    double x_econ[5];
    sparse_qr_solve(&qr_econ, b, x_econ, NULL);

    for (int i = 0; i < 5; i++)
        ASSERT_NEAR(x_full[i], x_econ[i], 1e-12);

    sparse_qr_free(&qr_full);
    sparse_qr_free(&qr_econ);
    sparse_free(A);
}

/* Economy R should be n×n upper triangular */
static void test_economy_r_shape(void) {
    idx_t m = 30, nc = 5;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if (i == j)
                sparse_insert(A, i, j, (double)(i + 1));

    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 1};
    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor_opts(A, &opts, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* R should have min(m,n)=nc rows */
    ASSERT_NOT_NULL(qr.R);
    if (qr.R) {
        ASSERT_EQ(sparse_rows(qr.R), nc);
        ASSERT_EQ(sparse_cols(qr.R), nc);
    }
    ASSERT_EQ(qr.rank, nc);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Rank-deficient tall matrix with economy */
static void test_economy_rank_deficient(void) {
    idx_t m = 20, nc = 4;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* col1 = col0, so rank = 3 at most */
    for (idx_t i = 0; i < m; i++) {
        sparse_insert(A, i, 0, (double)(i + 1));
        sparse_insert(A, i, 1, (double)(i + 1)); /* duplicate column */
        sparse_insert(A, i, 2, (double)(i * 2 + 1));
        sparse_insert(A, i, 3, (double)(i + 3));
    }

    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 1};
    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor_opts(A, &opts, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    printf("    economy rank-deficient 20x4: rank=%d\n", (int)qr.rank);
    ASSERT_TRUE(qr.rank < nc); /* should detect rank deficiency */

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Wide matrix (m < n) with economy=1: same as full */
static void test_economy_wide(void) {
    idx_t m = 3, nc = 6;
    SparseMatrix *A = sparse_create(m, nc);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < nc; j++)
            if (i == j || j == i + 3)
                sparse_insert(A, i, j, (double)(i + j + 1));

    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 1};
    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor_opts(A, &opts, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* For wide (m<n), economy Q is m×m (same as full) */
    ASSERT_EQ(qr.rank, m);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* 1×1 matrix with economy */
static void test_economy_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 7.0);

    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 1};
    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor_opts(A, &opts, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double b[1] = {14.0};
    double x[1];
    sparse_qr_solve(&qr, b, x, NULL);
    ASSERT_NEAR(x[0], 2.0, 1e-12);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* nos4 (square): economy identical to full */
static void test_economy_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(b);
    if (!b) {
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    /* Full QR */
    sparse_qr_t qr_full;
    sparse_err_t err = sparse_qr_factor(A, &qr_full);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(b);
        sparse_free(A);
        return;
    }
    double *x_full = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_full);
    if (!x_full) {
        free(b);
        sparse_qr_free(&qr_full);
        sparse_free(A);
        return;
    }
    sparse_qr_solve(&qr_full, b, x_full, NULL);

    /* Economy QR */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 1};
    sparse_qr_t qr_econ;
    err = sparse_qr_factor_opts(A, &opts, &qr_econ);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(x_full);
        free(b);
        sparse_qr_free(&qr_full);
        sparse_free(A);
        return;
    }
    double *x_econ = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_econ);
    if (!x_econ) {
        free(x_full);
        free(b);
        sparse_qr_free(&qr_full);
        sparse_qr_free(&qr_econ);
        sparse_free(A);
        return;
    }
    sparse_qr_solve(&qr_econ, b, x_econ, NULL);

    {
        double max_diff = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double d = fabs(x_full[i] - x_econ[i]);
            if (d > max_diff)
                max_diff = d;
        }
        printf("    nos4 economy vs full: max_diff=%.3e\n", max_diff);
        ASSERT_TRUE(max_diff < 1e-10);
    }

    free(x_full);
    free(x_econ);
    free(b);
    sparse_qr_free(&qr_full);
    sparse_qr_free(&qr_econ);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sparse-mode QR tests (Sprint 7 Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Sparse-mode QR matches dense-mode on small matrix */
static void test_sparse_mode_basic(void) {
    SparseMatrix *A = tf_qr_make_small_banded_4x3(1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    double b[4] = {1.0, 2.0, 3.0, 4.0};

    /* Dense-mode QR */
    sparse_qr_t qr_dense;
    sparse_err_t err = sparse_qr_factor(A, &qr_dense);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    double x_dense[3];
    double res_dense = 0.0;
    sparse_qr_solve(&qr_dense, b, x_dense, &res_dense);

    /* Sparse-mode QR */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 0, .sparse_mode = 1};
    sparse_qr_t qr_sparse;
    err = sparse_qr_factor_opts(A, &opts, &qr_sparse);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_qr_free(&qr_dense);
        sparse_free(A);
        return;
    }
    double x_sparse[3];
    double res_sparse = 0.0;
    sparse_qr_solve(&qr_sparse, b, x_sparse, &res_sparse);

    printf("    sparse-mode 4x3: dense_res=%.3e, sparse_res=%.3e\n", res_dense, res_sparse);

    /* Solutions should match */
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x_dense[i], x_sparse[i], 1e-10);
    ASSERT_NEAR(res_dense, res_sparse, 1e-10);
    ASSERT_EQ(qr_dense.rank, qr_sparse.rank);

    sparse_qr_free(&qr_dense);
    sparse_qr_free(&qr_sparse);
    sparse_free(A);
}

/* Sparse-mode QR on nos4 matches dense-mode */
static void test_sparse_mode_nos4(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    idx_t n = sparse_rows(A);

    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(b);
    if (!b) {
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    /* Dense-mode */
    sparse_qr_t qr_dense;
    sparse_err_t err = sparse_qr_factor(A, &qr_dense);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(b);
        sparse_free(A);
        return;
    }
    double *x_dense = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_dense);
    if (!x_dense) {
        free(b);
        sparse_qr_free(&qr_dense);
        sparse_free(A);
        return;
    }
    sparse_qr_solve(&qr_dense, b, x_dense, NULL);

    /* Sparse-mode */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 0, .sparse_mode = 1};
    sparse_qr_t qr_sparse;
    err = sparse_qr_factor_opts(A, &opts, &qr_sparse);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(x_dense);
        free(b);
        sparse_qr_free(&qr_dense);
        sparse_free(A);
        return;
    }
    double *x_sparse = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x_sparse);
    if (!x_sparse) {
        free(x_dense);
        free(b);
        sparse_qr_free(&qr_dense);
        sparse_qr_free(&qr_sparse);
        sparse_free(A);
        return;
    }
    sparse_qr_solve(&qr_sparse, b, x_sparse, NULL);

    {
        double max_diff = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double d = fabs(x_dense[i] - x_sparse[i]);
            if (d > max_diff)
                max_diff = d;
        }
        printf("    sparse-mode nos4: max_diff=%.3e, rank_dense=%d, rank_sparse=%d\n", max_diff,
               (int)qr_dense.rank, (int)qr_sparse.rank);
        ASSERT_TRUE(max_diff < 1e-8);
    }

    free(x_dense);
    free(x_sparse);
    free(b);
    sparse_qr_free(&qr_dense);
    sparse_qr_free(&qr_sparse);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sparse-mode QR hardening (Sprint 7 Day 9)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: compare dense and sparse QR solve on a given matrix */
static void compare_dense_sparse_qr(const SparseMatrix *A, const char *name) {
    idx_t m = sparse_rows(A);
    idx_t nc = sparse_cols(A);

    double *b = malloc((size_t)m * sizeof(double));
    ASSERT_NOT_NULL(b);
    if (!b)
        return;
    for (idx_t i = 0; i < m; i++)
        b[i] = (double)(i + 1);

    /* Dense-mode QR */
    sparse_qr_t qr_d;
    sparse_err_t err = sparse_qr_factor(A, &qr_d);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(b);
        return;
    }
    double *x_d = malloc((size_t)nc * sizeof(double));
    ASSERT_NOT_NULL(x_d);
    if (!x_d) {
        free(b);
        sparse_qr_free(&qr_d);
        return;
    }
    double res_d = 0.0;
    sparse_qr_solve(&qr_d, b, x_d, &res_d);

    /* Sparse-mode QR */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 0, .sparse_mode = 1};
    sparse_qr_t qr_s;
    err = sparse_qr_factor_opts(A, &opts, &qr_s);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(x_d);
        free(b);
        sparse_qr_free(&qr_d);
        return;
    }
    double *x_s = malloc((size_t)nc * sizeof(double));
    ASSERT_NOT_NULL(x_s);
    if (!x_s) {
        free(x_d);
        free(b);
        sparse_qr_free(&qr_d);
        sparse_qr_free(&qr_s);
        return;
    }
    double res_s = 0.0;
    sparse_qr_solve(&qr_s, b, x_s, &res_s);

    {
        double max_diff = 0.0;
        for (idx_t i = 0; i < nc; i++) {
            double d = fabs(x_d[i] - x_s[i]);
            if (d > max_diff)
                max_diff = d;
        }
        printf("    sparse vs dense %s: max_diff=%.3e, rank_d=%d, rank_s=%d\n", name, max_diff,
               (int)qr_d.rank, (int)qr_s.rank);
        ASSERT_TRUE(max_diff < 1e-8);
        ASSERT_EQ(qr_d.rank, qr_s.rank);
    }

    free(x_d);
    free(x_s);
    free(b);
    sparse_qr_free(&qr_d);
    sparse_qr_free(&qr_s);
}

/* Sparse-mode: tall-skinny 50×10 */
static void test_sparse_mode_tall(void) {
    idx_t m = 50, nc = 10;
    SparseMatrix *A = tf_qr_make_tall_diagonal_dominant(m, nc, 10.0, 1.0, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    compare_dense_sparse_qr(A, "50x10");
    sparse_free(A);
}

/* Sparse-mode: wide matrix 3×6 */
static void test_sparse_mode_wide(void) {
    SparseMatrix *A = sparse_create(3, 6);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 0, 4, 2.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 1, 5, 3.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 2, 5.0);
    sparse_insert(A, 2, 4, 1.0);
    compare_dense_sparse_qr(A, "3x6");
    sparse_free(A);
}

/* Sparse-mode: rank-deficient 4×3 with duplicate column */
static void test_sparse_mode_rank_deficient(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* col1 = col0 → rank 2 */
    for (idx_t i = 0; i < 4; i++) {
        sparse_insert(A, i, 0, (double)(i + 1));
        sparse_insert(A, i, 1, (double)(i + 1));
        sparse_insert(A, i, 2, (double)(i * 2 + 1));
    }
    compare_dense_sparse_qr(A, "rank-def 4x3");
    sparse_free(A);
}

/* Sparse-mode: west0067 */
static void test_sparse_mode_west0067(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/west0067.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    compare_dense_sparse_qr(A, "west0067");
    sparse_free(A);
}

/* Sparse-mode: bcsstk04 */
static void test_sparse_mode_bcsstk04(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;
    compare_dense_sparse_qr(A, "bcsstk04");
    sparse_free(A);
}

/* Sparse-mode: 1×1 */
static void test_sparse_mode_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 7.0);
    compare_dense_sparse_qr(A, "1x1");
    sparse_free(A);
}

/* Sparse-mode: Q orthogonality */
static void test_sparse_mode_q_ortho(void) {
    SparseMatrix *A = sparse_create(5, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 5.0);
    sparse_insert(A, 3, 0, 2.0);
    sparse_insert(A, 3, 2, 1.0);
    sparse_insert(A, 4, 1, 2.0);

    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 0, .sparse_mode = 1};
    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor_opts(A, &opts, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    /* Form Q and check orthogonality */
    idx_t m = 5;
    double *Q = calloc((size_t)m * (size_t)m, sizeof(double));
    ASSERT_NOT_NULL(Q);
    if (!Q) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    sparse_qr_form_q(&qr, Q);

    double max_err = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < m; j++) {
            double dot = 0.0;
            for (idx_t p = 0; p < m; p++)
                dot += Q[(size_t)i * (size_t)m + (size_t)p] * Q[(size_t)j * (size_t)m + (size_t)p];
            double expected = (i == j) ? 1.0 : 0.0;
            double e = fabs(dot - expected);
            if (e > max_err)
                max_err = e;
        }
    }
    printf("    sparse-mode Q ortho: max_err=%.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-10);

    free(Q);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sparse-mode QR: benchmarks & edge cases (Sprint 7 Day 10)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Reconstruction error comparison: sparse vs dense on nos4 */
static void test_sparse_mode_reconstruction(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    /* Dense-mode reconstruction error */
    sparse_qr_t qr_d;
    sparse_err_t err = sparse_qr_factor(A, &qr_d);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    double recon_d = tf_qr_reconstruction_max_error(A, &qr_d);

    /* Sparse-mode reconstruction error */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .economy = 0, .sparse_mode = 1};
    sparse_qr_t qr_s;
    err = sparse_qr_factor_opts(A, &opts, &qr_s);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_qr_free(&qr_d);
        sparse_free(A);
        return;
    }
    double recon_s = tf_qr_reconstruction_max_error(A, &qr_s);

    printf("    reconstruction nos4: dense=%.3e, sparse=%.3e\n", recon_d, recon_s);
    ASSERT_TRUE(recon_d < 1e-10);
    ASSERT_TRUE(recon_s < 1e-10);

    sparse_qr_free(&qr_d);
    sparse_qr_free(&qr_s);
    sparse_free(A);
}

/* Sparse-mode with AMD reordering */
static void test_sparse_mode_amd(void) {
    SparseMatrix *A = sparse_create(6, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 2, 1.0);
    sparse_insert(A, 3, 3, 4.0);
    sparse_insert(A, 4, 0, 2.0);
    sparse_insert(A, 5, 3, 2.0);

    double b[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    /* Sparse-mode without reorder */
    sparse_qr_opts_t opts_none = {.reorder = SPARSE_REORDER_NONE, .sparse_mode = 1};
    sparse_qr_t qr_none;
    sparse_err_t err = sparse_qr_factor_opts(A, &opts_none, &qr_none);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    double x_none[4];
    sparse_qr_solve(&qr_none, b, x_none, NULL);

    /* Sparse-mode with AMD reorder */
    sparse_qr_opts_t opts_amd = {.reorder = SPARSE_REORDER_AMD, .sparse_mode = 1};
    sparse_qr_t qr_amd;
    err = sparse_qr_factor_opts(A, &opts_amd, &qr_amd);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_qr_free(&qr_none);
        sparse_free(A);
        return;
    }
    double x_amd[4];
    sparse_qr_solve(&qr_amd, b, x_amd, NULL);

    /* Solutions should agree */
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x_none[i], x_amd[i], 1e-10);

    printf("    sparse AMD reorder: rank_none=%d, rank_amd=%d\n", (int)qr_none.rank,
           (int)qr_amd.rank);

    sparse_qr_free(&qr_none);
    sparse_qr_free(&qr_amd);
    sparse_free(A);
}

/* Very sparse matrix: diagonal only */
static void test_sparse_mode_diagonal(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    compare_dense_sparse_qr(A, "diag 20x20");
    sparse_free(A);
}

/* Single column matrix */
static void test_sparse_mode_single_col(void) {
    SparseMatrix *A = sparse_create(5, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 2, 0, 4.0);
    compare_dense_sparse_qr(A, "5x1");
    sparse_free(A);
}

/* Single row matrix */
static void test_sparse_mode_single_row(void) {
    SparseMatrix *A = sparse_create(1, 5);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 0, 3, 5.0);
    compare_dense_sparse_qr(A, "1x5");
    sparse_free(A);
}

/* Timing comparison: print factorization time for dense vs sparse on nos4 */
static void test_sparse_mode_timing(void) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    ASSERT_ERR(lerr, SPARSE_OK);
    if (lerr != SPARSE_OK || !A)
        return;

    /* Dense-mode */
    sparse_qr_t qr_d;
    sparse_err_t err_d = sparse_qr_factor(A, &qr_d);
    ASSERT_ERR(err_d, SPARSE_OK);
    if (err_d != SPARSE_OK) {
        sparse_free(A);
        return;
    }
    idx_t nnz_r_d = sparse_nnz(qr_d.R);

    /* Sparse-mode */
    sparse_qr_opts_t opts = {.reorder = SPARSE_REORDER_NONE, .sparse_mode = 1};
    sparse_qr_t qr_s;
    sparse_err_t err_s = sparse_qr_factor_opts(A, &opts, &qr_s);
    ASSERT_ERR(err_s, SPARSE_OK);
    if (err_s != SPARSE_OK) {
        sparse_qr_free(&qr_d);
        sparse_free(A);
        return;
    }
    idx_t nnz_r_s = sparse_nnz(qr_s.R);

    printf("    nos4 R nnz: dense=%d, sparse=%d\n", (int)nnz_r_d, (int)nnz_r_s);
    /* Both should produce nearly same R fill-in.  Relative drop tolerance
     * may cause ±1 borderline entry difference between modes. */
    ASSERT_TRUE(abs(nnz_r_d - nnz_r_s) <= 1);

    sparse_qr_free(&qr_d);
    sparse_qr_free(&qr_s);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * QR iterative refinement tests (Sprint 7 Day 11)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Well-conditioned: refinement should barely change solution */
static void test_qr_refine_well_conditioned(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 10.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 1, 10.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 2, 10.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 3, 2, 1.0);

    double b[4] = {11.0, 11.0, 11.0, 3.0};

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[3];
    double res_before = 0.0;
    sparse_qr_solve(&qr, b, x, &res_before);

    double res_after = 0.0;
    ASSERT_ERR(sparse_qr_refine(&qr, A, b, x, 3, &res_after), SPARSE_OK);

    printf("    well-cond refine: before=%.3e, after=%.3e\n", res_before, res_after);
    ASSERT_TRUE(res_after <= res_before + 1e-15);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Ill-conditioned: refinement should improve residual */
static void test_qr_refine_ill_conditioned(void) {
    SparseMatrix *A = sparse_create(4, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Near-singular: col2 ~= col0 + small perturbation */
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 0, 2, 1.0 + 1e-8);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, 3.0 + 1e-8);
    sparse_insert(A, 2, 0, 5.0);
    sparse_insert(A, 2, 1, 6.0);
    sparse_insert(A, 2, 2, 5.0 + 1e-8);
    sparse_insert(A, 3, 0, 7.0);
    sparse_insert(A, 3, 1, 8.0);
    sparse_insert(A, 3, 2, 7.0 + 1e-8);

    double b[4] = {1.0, 2.0, 3.0, 4.0};

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[3];
    double res_before = 0.0;
    sparse_qr_solve(&qr, b, x, &res_before);

    double res_after = 0.0;
    ASSERT_ERR(sparse_qr_refine(&qr, A, b, x, 5, &res_after), SPARSE_OK);

    printf("    ill-cond refine: before=%.3e, after=%.3e\n", res_before, res_after);
    /* Refinement should not make things worse */
    ASSERT_TRUE(res_after <= res_before + 1e-12);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* max_refine=0: just computes residual, doesn't modify x */
static void test_qr_refine_zero_iter(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 4.0);

    double b[3] = {2.0, 6.0, 12.0};

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        sparse_free(A);
        return;
    }

    double x[3];
    sparse_qr_solve(&qr, b, x, NULL);

    double x_copy[3] = {x[0], x[1], x[2]};
    double res = 0.0;
    ASSERT_ERR(sparse_qr_refine(&qr, A, b, x, 0, &res), SPARSE_OK);

    /* x should be unchanged */
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_copy[i], 1e-15);
    /* residual should be computed */
    printf("    zero-iter refine: res=%.3e\n", res);
    ASSERT_TRUE(res < 1e-12);

    sparse_qr_free(&qr);
    sparse_free(A);
}

/* nos4: QR solve + refine */
static void test_qr_refine_nos4(void) {
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
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(x_exact);
        free(b);
        sparse_free(A);
        return;
    }

    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    if (!x) {
        free(x_exact);
        free(b);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double res_before = 0.0;
    sparse_qr_solve(&qr, b, x, &res_before);

    double res_after = 0.0;
    ASSERT_ERR(sparse_qr_refine(&qr, A, b, x, 3, &res_after), SPARSE_OK);

    printf("    nos4 refine: before=%.3e, after=%.3e\n", res_before, res_after);
    /* Refinement may introduce tiny rounding noise on already-exact solutions */
    ASSERT_TRUE(res_after < 1e-10);

    free(x_exact);
    free(b);
    free(x);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* Overdetermined least-squares: refinement on tall system */
static void test_qr_refine_overdetermined(void) {
    idx_t m = 20, nc = 5;
    SparseMatrix *A = tf_qr_make_tall_diagonal_dominant(m, nc, 5.0, 1.0, 1);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    double *b = malloc((size_t)m * sizeof(double));
    ASSERT_NOT_NULL(b);
    if (!b) {
        sparse_free(A);
        return;
    }
    for (idx_t i = 0; i < m; i++)
        b[i] = (double)(i + 1);

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK) {
        free(b);
        sparse_free(A);
        return;
    }

    double *x = malloc((size_t)nc * sizeof(double));
    ASSERT_NOT_NULL(x);
    if (!x) {
        free(b);
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double res_before = 0.0;
    sparse_qr_solve(&qr, b, x, &res_before);

    double res_after = 0.0;
    ASSERT_ERR(sparse_qr_refine(&qr, A, b, x, 3, &res_after), SPARSE_OK);

    printf("    overdetermined refine: before=%.3e, after=%.3e\n", res_before, res_after);
    ASSERT_TRUE(res_after <= res_before + 1e-12);

    free(b);
    free(x);
    sparse_qr_free(&qr);
    sparse_free(A);
}

/* NULL input handling */
static void test_qr_refine_null(void) {
    ASSERT_ERR(sparse_qr_refine(NULL, NULL, NULL, NULL, 0, NULL), SPARSE_ERR_NULL);
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
    RUN_TEST(test_qr_public_scalar_alias);
    RUN_TEST(test_qr_rejects_factored_matrix_reuse);
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

    /* Rank estimation and null space (Day 10) */
    RUN_TEST(test_rank_full);
    RUN_TEST(test_rank_1_nullspace);
    RUN_TEST(test_known_nullspace);
    RUN_TEST(test_qr_external_dense_reference_rank1_4x3_nullspace_projector);
    RUN_TEST(test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector);
    RUN_TEST(test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector);
    RUN_TEST(test_rank_rect_deficient);
    RUN_TEST(test_rank_explicit_tol);
    RUN_TEST(test_qr_rank_dependent_row_fixture);
    RUN_TEST(test_qr_rank_diagonal_threshold_fixture);
    RUN_TEST(test_qr_external_dense_reference_rank_threshold_diag4_family);
    RUN_TEST(test_qr_external_dense_reference_rank_threshold_diag4_scaled_family);
    RUN_TEST(test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family);

    /* Column reordering (Day 11) */
    RUN_TEST(test_qr_reorder_amd_solve);
    RUN_TEST(test_qr_reorder_nos4_fillin);
    RUN_TEST(test_qr_reorder_none);

    /* Economy QR (Sprint 7 Day 7) */
    RUN_TEST(test_economy_solve_tall);
    RUN_TEST(test_economy_q_orthogonality);
    RUN_TEST(test_qr_external_dense_reference_economy_projector_5x3);
    RUN_TEST(test_economy_square);
    RUN_TEST(test_economy_r_shape);
    RUN_TEST(test_economy_rank_deficient);
    RUN_TEST(test_economy_wide);
    RUN_TEST(test_economy_1x1);
    RUN_TEST(test_economy_nos4);

    /* Sparse-mode QR (Sprint 7 Day 8) */
    RUN_TEST(test_sparse_mode_basic);
    RUN_TEST(test_sparse_mode_nos4);

    /* Sparse-mode hardening (Sprint 7 Day 9) */
    RUN_TEST(test_sparse_mode_tall);
    RUN_TEST(test_sparse_mode_wide);
    RUN_TEST(test_sparse_mode_rank_deficient);
    RUN_TEST(test_sparse_mode_west0067);
    RUN_TEST(test_sparse_mode_bcsstk04);
    RUN_TEST(test_sparse_mode_1x1);
    RUN_TEST(test_sparse_mode_q_ortho);

    /* Sparse-mode benchmarks & edge cases (Sprint 7 Day 10) */
    RUN_TEST(test_sparse_mode_reconstruction);
    RUN_TEST(test_sparse_mode_amd);
    RUN_TEST(test_sparse_mode_diagonal);
    RUN_TEST(test_sparse_mode_single_col);
    RUN_TEST(test_sparse_mode_single_row);
    RUN_TEST(test_sparse_mode_timing);

    /* QR iterative refinement (Sprint 7 Day 11) */
    RUN_TEST(test_qr_refine_well_conditioned);
    RUN_TEST(test_qr_refine_ill_conditioned);
    RUN_TEST(test_qr_refine_zero_iter);
    RUN_TEST(test_qr_refine_nos4);
    RUN_TEST(test_qr_refine_overdetermined);
    RUN_TEST(test_qr_refine_null);

    TEST_SUITE_END();
}
