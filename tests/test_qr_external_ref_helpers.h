#ifndef TEST_QR_EXTERNAL_REF_HELPERS_H
#define TEST_QR_EXTERNAL_REF_HELPERS_H

/* Sprint 193 QR external-reference helper boundary.
 * Owns the selected rank/nullspace/threshold dense-reference helpers while
 * tests/test_qr.c keeps the test_qr proof-owner registration.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef TF_ENABLE_EXTERNAL_REFERENCE_HELPER
#define TF_ENABLE_EXTERNAL_REFERENCE_HELPER
#endif
#include "test_qr_helpers.h"
#include "test_solver_helpers.h"

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
        strcmp(fixture_key, "qr_rankdef_wide_3x5_nullspace_subspace") != 0 &&
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
        strcmp(fixture_key, "qr_rank_threshold_duplicate_5x4_perturbed_family") != 0 &&
        strcmp(fixture_key, "qr_rank_threshold_dependent_row_4x3_perturbed_family") != 0) {
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

static void test_qr_external_reference_readers_reject_invalid_arguments(void) {
    double values[4] = {0.0};
    char reason[128] = {0};

    ASSERT_EQ(read_qr_basis_external_reference(NULL, values, 4, reason, sizeof(reason)),
              TF_EXTERNAL_REFERENCE_ERROR);
    ASSERT_NOT_NULL(strstr(reason, "external QR basis reference invalid arguments"));

    reason[0] = '\0';
    ASSERT_EQ(read_qr_basis_external_reference("qr_rank1_4x3_nullspace_projector", NULL, 4, reason,
                                               sizeof(reason)),
              TF_EXTERNAL_REFERENCE_ERROR);
    ASSERT_NOT_NULL(strstr(reason, "external QR basis reference invalid arguments"));

    reason[0] = '\0';
    ASSERT_EQ(read_qr_threshold_external_reference(NULL, values, 4, reason, sizeof(reason)),
              TF_EXTERNAL_REFERENCE_ERROR);
    ASSERT_NOT_NULL(strstr(reason, "external QR threshold reference invalid arguments"));

    reason[0] = '\0';
    ASSERT_EQ(read_qr_threshold_external_reference("qr_rank_threshold_diag4_family", NULL, 4,
                                                   reason, sizeof(reason)),
              TF_EXTERNAL_REFERENCE_ERROR);
    ASSERT_NOT_NULL(strstr(reason, "external QR threshold reference invalid arguments"));

    ASSERT_EQ(read_qr_basis_external_reference("qr_rank1_4x3_nullspace_projector", values, 4, NULL,
                                               sizeof(reason)),
              TF_EXTERNAL_REFERENCE_ERROR);
    ASSERT_EQ(read_qr_threshold_external_reference("qr_rank_threshold_diag4_family", values, 4,
                                                   NULL, sizeof(reason)),
              TF_EXTERNAL_REFERENCE_ERROR);
}

static void test_qr_external_reference_readers_reject_unsupported_fixtures(void) {
    double values[4] = {0.0};
    char reason[128] = {0};

    ASSERT_EQ(read_qr_basis_external_reference("qr_unknown_basis_fixture", values, 4, reason,
                                               sizeof(reason)),
              TF_EXTERNAL_REFERENCE_ERROR);
    ASSERT_NOT_NULL(strstr(reason, "unsupported external QR basis reference fixture key"));
    ASSERT_NOT_NULL(strstr(reason, "qr_unknown_basis_fixture"));

    reason[0] = '\0';
    ASSERT_EQ(read_qr_threshold_external_reference("qr_unknown_threshold_fixture", values, 4,
                                                   reason, sizeof(reason)),
              TF_EXTERNAL_REFERENCE_ERROR);
    ASSERT_NOT_NULL(strstr(reason, "unsupported external QR threshold reference fixture key"));
    ASSERT_NOT_NULL(strstr(reason, "qr_unknown_threshold_fixture"));
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
            (4 + QR_DEPENDENT_ROW_NULLSPACE_N * QR_DEPENDENT_ROW_NULLSPACE_N)
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
    ASSERT_TRUE(ref[3] <= 0.0);
    if ((idx_t)ref[0] != QR_DEPENDENT_ROW_NULLSPACE_N || (idx_t)ref[1] != 2 || (idx_t)ref[2] != 1 ||
        ref[3] > 0.0)
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
    ASSERT_EQ(ndim, 1);
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

static SparseMatrix *make_rankdef_wide_3x5(void) {
    SparseMatrix *A = sparse_create(3, 5);
    if (!A)
        return NULL;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 0, 1, 2.0) ||
        !tf_qr_insert_or_free(&A, 0, 3, 1.0) || !tf_qr_insert_or_free(&A, 1, 1, 3.0) ||
        !tf_qr_insert_or_free(&A, 1, 2, 1.0) || !tf_qr_insert_or_free(&A, 1, 4, 2.0) ||
        !tf_qr_insert_or_free(&A, 2, 0, 1.0) || !tf_qr_insert_or_free(&A, 2, 1, 5.0) ||
        !tf_qr_insert_or_free(&A, 2, 2, 1.0) || !tf_qr_insert_or_free(&A, 2, 3, 1.0) ||
        !tf_qr_insert_or_free(&A, 2, 4, 2.0))
        return NULL;
    return A;
}

static void test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace(void) {
#ifdef _WIN32
    SKIP_TEST("external QR dense reference helper is not enabled on Windows");
#else
    enum { QR_WIDE_NULLSPACE_N = 5 };
    enum { QR_WIDE_NULLSPACE_NULLITY = 3 };
    enum { QR_WIDE_NULLSPACE_PROJECTOR_VALUES = 4 + QR_WIDE_NULLSPACE_N * QR_WIDE_NULLSPACE_N };
    double ref[QR_WIDE_NULLSPACE_PROJECTOR_VALUES] = {0.0};
    char reason[256] = {0};
    int ref_status = read_qr_basis_external_reference("qr_rankdef_wide_3x5_nullspace_subspace", ref,
                                                      QR_WIDE_NULLSPACE_PROJECTOR_VALUES, reason,
                                                      sizeof(reason));
    if (ref_status == TF_EXTERNAL_REFERENCE_SKIP)
        SKIP_TEST(reason);
    if (ref_status != TF_EXTERNAL_REFERENCE_OK) {
        TF_FAIL_("external QR basis reference failed: %s", reason);
        return;
    }

    ASSERT_EQ((idx_t)ref[0], QR_WIDE_NULLSPACE_N);
    ASSERT_EQ((idx_t)ref[1], 2);
    ASSERT_EQ((idx_t)ref[2], QR_WIDE_NULLSPACE_NULLITY);
    ASSERT_NEAR(ref[3], 0.0, 0.0);
    if ((idx_t)ref[0] != QR_WIDE_NULLSPACE_N || (idx_t)ref[1] != 2 ||
        (idx_t)ref[2] != QR_WIDE_NULLSPACE_NULLITY || ref[3] != 0.0)
        return;

    SparseMatrix *A = make_rankdef_wide_3x5();
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
    ASSERT_EQ(ndim, QR_WIDE_NULLSPACE_NULLITY);
    if (qr.rank != 2 || ndim != QR_WIDE_NULLSPACE_NULLITY) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double basis[QR_WIDE_NULLSPACE_N * QR_WIDE_NULLSPACE_NULLITY] = {0.0};
    sparse_err_t basis_err = sparse_qr_nullspace(&qr, ref[3], basis, &ndim);
    ASSERT_ERR(basis_err, SPARSE_OK);
    if (basis_err != SPARSE_OK) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double z[QR_WIDE_NULLSPACE_N * QR_WIDE_NULLSPACE_NULLITY] = {0.0};
    for (idx_t col = 0; col < QR_WIDE_NULLSPACE_NULLITY; col++) {
        for (idx_t row = 0; row < QR_WIDE_NULLSPACE_N; row++)
            z[(size_t)col * QR_WIDE_NULLSPACE_N + (size_t)row] =
                basis[(size_t)col * QR_WIDE_NULLSPACE_N + (size_t)row];
        for (idx_t prev = 0; prev < col; prev++) {
            double dot = 0.0;
            for (idx_t row = 0; row < QR_WIDE_NULLSPACE_N; row++)
                dot += z[(size_t)col * QR_WIDE_NULLSPACE_N + (size_t)row] *
                       z[(size_t)prev * QR_WIDE_NULLSPACE_N + (size_t)row];
            for (idx_t row = 0; row < QR_WIDE_NULLSPACE_N; row++)
                z[(size_t)col * QR_WIDE_NULLSPACE_N + (size_t)row] -=
                    dot * z[(size_t)prev * QR_WIDE_NULLSPACE_N + (size_t)row];
        }
        double norm = vec_norm2(&z[(size_t)col * QR_WIDE_NULLSPACE_N], QR_WIDE_NULLSPACE_N);
        ASSERT_TRUE(norm > 0.0);
        if (norm <= 0.0) {
            sparse_qr_free(&qr);
            sparse_free(A);
            return;
        }
        for (idx_t row = 0; row < QR_WIDE_NULLSPACE_N; row++)
            z[(size_t)col * QR_WIDE_NULLSPACE_N + (size_t)row] /= norm;
    }

    double max_null_residual = 0.0;
    for (idx_t col = 0; col < QR_WIDE_NULLSPACE_NULLITY; col++) {
        double Av[3] = {0.0};
        sparse_matvec(A, &z[(size_t)col * QR_WIDE_NULLSPACE_N], Av);
        double residual = vec_norm2(Av, 3);
        if (residual > max_null_residual)
            max_null_residual = residual;
    }

    double max_orthogonality_err = 0.0;
    for (idx_t col_i = 0; col_i < QR_WIDE_NULLSPACE_NULLITY; col_i++) {
        for (idx_t col_j = 0; col_j < QR_WIDE_NULLSPACE_NULLITY; col_j++) {
            double dot = 0.0;
            for (idx_t row = 0; row < QR_WIDE_NULLSPACE_N; row++) {
                dot += z[(size_t)col_i * QR_WIDE_NULLSPACE_N + (size_t)row] *
                       z[(size_t)col_j * QR_WIDE_NULLSPACE_N + (size_t)row];
            }
            double expected = col_i == col_j ? 1.0 : 0.0;
            double diff = fabs(dot - expected);
            if (diff > max_orthogonality_err)
                max_orthogonality_err = diff;
        }
    }

    double max_projector_diff = 0.0;
    for (idx_t row = 0; row < QR_WIDE_NULLSPACE_N; row++) {
        for (idx_t col = 0; col < QR_WIDE_NULLSPACE_N; col++) {
            double product = 0.0;
            for (idx_t k = 0; k < QR_WIDE_NULLSPACE_NULLITY; k++)
                product += z[(size_t)k * QR_WIDE_NULLSPACE_N + (size_t)row] *
                           z[(size_t)k * QR_WIDE_NULLSPACE_N + (size_t)col];
            double diff = fabs(product - ref[4 + (size_t)col * QR_WIDE_NULLSPACE_N + (size_t)row]);
            if (diff > max_projector_diff)
                max_projector_diff = diff;
        }
    }

    printf("    external QR dense ref rankdef_wide_3x5_nullspace_subspace: "
           "projector diff = %.3e, null residual = %.3e, orthogonality err = %.3e\n",
           max_projector_diff, max_null_residual, max_orthogonality_err);
    ASSERT_TRUE(max_projector_diff < 1e-8);
    ASSERT_TRUE(max_null_residual < 1e-10);
    ASSERT_TRUE(max_orthogonality_err < 1e-10);

    sparse_qr_free(&qr);
    sparse_free(A);
#endif
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
            (QR_THRESHOLD_PERTURBED_RECORDS * QR_THRESHOLD_PERTURBED_FIELDS)
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

static void
test_qr_external_dense_reference_rank_threshold_dependent_row_4x3_perturbed_family(void) {
#ifdef _WIN32
    SKIP_TEST("external QR dense reference helper is not enabled on Windows");
#else
    enum { QR_THRESHOLD_DEP_ROW_RECORDS = 3 };
    enum { QR_THRESHOLD_DEP_ROW_FIELDS = 3 };
    enum {
        QR_THRESHOLD_DEP_ROW_VALUES = QR_THRESHOLD_DEP_ROW_RECORDS * QR_THRESHOLD_DEP_ROW_FIELDS
    };
    double ref[QR_THRESHOLD_DEP_ROW_VALUES] = {0.0};
    char reason[256] = {0};
    int ref_status = read_qr_threshold_external_reference(
        "qr_rank_threshold_dependent_row_4x3_perturbed_family", ref, QR_THRESHOLD_DEP_ROW_VALUES,
        reason, sizeof(reason));
    if (ref_status == TF_EXTERNAL_REFERENCE_SKIP)
        SKIP_TEST(reason);
    if (ref_status != TF_EXTERNAL_REFERENCE_OK) {
        TF_FAIL_("external QR threshold reference failed: %s", reason);
        return;
    }

    const double expected_perturbation = 1e-6;
    const double expected_thresholds[QR_THRESHOLD_DEP_ROW_RECORDS] = {1e-10, 1e-8, 1e-6};
    const idx_t expected_ranks[QR_THRESHOLD_DEP_ROW_RECORDS] = {3, 3, 2};

    SparseMatrix *A = tf_qr_make_dependent_row_4x3();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    sparse_err_t insert_err = sparse_insert(A, 2, 2, 3.0 + expected_perturbation);
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

    double rdiag[3] = {0.0};
    sparse_err_t diag_err = sparse_qr_diag_r(&qr, rdiag);
    ASSERT_ERR(diag_err, SPARSE_OK);
    if (diag_err != SPARSE_OK) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }
    double r00 = fabs(rdiag[0]);
    double pivot_ratio = r00 > 0.0 ? fabs(rdiag[2]) / r00 : 0.0;

    for (idx_t i = 0; i < QR_THRESHOLD_DEP_ROW_RECORDS; i++) {
        double perturbation = ref[(size_t)i * QR_THRESHOLD_DEP_ROW_FIELDS];
        double threshold = ref[(size_t)i * QR_THRESHOLD_DEP_ROW_FIELDS + 1u];
        idx_t expected_rank = (idx_t)ref[(size_t)i * QR_THRESHOLD_DEP_ROW_FIELDS + 2u];
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
        printf("    external QR dense ref rank_threshold_dependent_row_4x3_perturbed_family: "
               "perturb=%.1e tol=%.0e abs_tol=%.3e expected=%d product=%d info=%d "
               "pivot_ratio=%.3e |Rdiag|=[%.3e, %.3e, %.3e]\n",
               perturbation, threshold, abs_threshold, (int)expected_rank, (int)product_rank,
               (int)info.rank, pivot_ratio, fabs(rdiag[0]), fabs(rdiag[1]), fabs(rdiag[2]));
        ASSERT_EQ(product_rank, expected_rank);
        ASSERT_EQ(info.rank, expected_rank);
    }

    sparse_qr_free(&qr);
    sparse_free(A);
#endif
}

#endif
