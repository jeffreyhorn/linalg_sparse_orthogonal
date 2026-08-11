#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "test_framework.h"
#include "test_qr_helpers.h"

#include <math.h>

#define QR_CORPUS_FIXTURE_KEY "qr_rank_deficient_6x4_nullspace_v1"
#define QR_CORPUS_RESIDUAL_TOL 1e-10
#define QR_CORPUS_REFERENCE_RESIDUAL_TOL 1e-12
#define QR_CORPUS_PROJECTOR_TOL 1e-8
#define QR_CORPUS_MINNORM_TOL 1e-10
#define QR_CORPUS_MAX_ROWS 5
#define QR_CORPUS_MAX_COLS 10

static void qr_corpus_assert_shape(const char *fixture_key, const SparseMatrix *A,
                                   idx_t expected_rows, idx_t expected_cols, idx_t expected_nnz) {
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    printf("    %s shape: rows=%d cols=%d nnz=%d\n", fixture_key, (int)sparse_rows(A),
           (int)sparse_cols(A), (int)sparse_nnz(A));
    ASSERT_EQ(sparse_rows(A), expected_rows);
    ASSERT_EQ(sparse_cols(A), expected_cols);
    ASSERT_EQ(sparse_nnz(A), expected_nnz);
}

static double qr_corpus_projector_distance(const double *basis, const double *reference,
                                           idx_t count) {
    double basis_norm_sq = 0.0;
    double reference_norm_sq = 0.0;
    for (idx_t i = 0; i < count; i++) {
        basis_norm_sq += basis[i] * basis[i];
        reference_norm_sq += reference[i] * reference[i];
    }
    ASSERT_TRUE(basis_norm_sq > 0.0);
    ASSERT_TRUE(reference_norm_sq > 0.0);
    if (basis_norm_sq <= 0.0 || reference_norm_sq <= 0.0)
        return HUGE_VAL;

    double max_diff = 0.0;
    for (idx_t row = 0; row < count; row++) {
        for (idx_t col = 0; col < count; col++) {
            double observed = basis[row] * basis[col] / basis_norm_sq;
            double expected = reference[row] * reference[col] / reference_norm_sq;
            double diff = fabs(observed - expected);
            if (diff > max_diff)
                max_diff = diff;
        }
    }
    return max_diff;
}

static void qr_corpus_assert_rankdef_fixture(const char *fixture_key, SparseMatrix *A,
                                             idx_t expected_rank, idx_t expected_nullity,
                                             const double *reference_null, idx_t cols) {
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    ASSERT_TRUE(cols <= QR_CORPUS_MAX_COLS);
    if (cols > QR_CORPUS_MAX_COLS)
        return;

    sparse_qr_t qr;
    sparse_err_t err = sparse_qr_factor(A, &qr);
    ASSERT_ERR(err, SPARSE_OK);
    if (err != SPARSE_OK)
        return;

    idx_t rank = sparse_qr_rank(&qr, 0.0);
    ASSERT_EQ(rank, expected_rank);

    idx_t nullity = -1;
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, NULL, &nullity), SPARSE_OK);
    ASSERT_EQ(nullity, expected_nullity);
    if (nullity != expected_nullity) {
        sparse_qr_free(&qr);
        return;
    }

    double basis[QR_CORPUS_MAX_COLS] = {0.0};
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, basis, &nullity), SPARSE_OK);
    ASSERT_EQ(nullity, expected_nullity);

    double residual = tf_qr_normalized_matvec_residual(A, basis, cols);
    double projector_distance = qr_corpus_projector_distance(basis, reference_null, cols);
    double reference_residual = tf_qr_normalized_matvec_residual(A, reference_null, cols);
    printf("    %s rank=%d nullity=%d residual=%.3e projector=%.3e ref_residual=%.3e\n",
           fixture_key, (int)rank, (int)nullity, residual, projector_distance, reference_residual);
    ASSERT_TRUE(residual <= QR_CORPUS_RESIDUAL_TOL);
    ASSERT_TRUE(projector_distance <= QR_CORPUS_PROJECTOR_TOL);
    ASSERT_TRUE(reference_residual <= QR_CORPUS_REFERENCE_RESIDUAL_TOL);

    sparse_qr_free(&qr);
}

static double qr_corpus_minnorm_residual(const SparseMatrix *A, const double *x, const double *b,
                                         idx_t rows) {
    ASSERT_TRUE(rows <= QR_CORPUS_MAX_ROWS);
    if (rows > QR_CORPUS_MAX_ROWS)
        return HUGE_VAL;

    double ax[QR_CORPUS_MAX_ROWS] = {0.0};
    sparse_err_t mv_err = sparse_matvec(A, x, ax);
    ASSERT_ERR(mv_err, SPARSE_OK);
    if (mv_err != SPARSE_OK)
        return HUGE_VAL;

    double residual_sq = 0.0;
    for (idx_t row = 0; row < rows; row++) {
        double diff = ax[row] - b[row];
        residual_sq += diff * diff;
    }
    return sqrt(residual_sq);
}

static void qr_corpus_assert_minnorm_fixture(const char *fixture_key, SparseMatrix *A,
                                             const double *b, const double *expected_x, idx_t rows,
                                             idx_t cols, double expected_norm) {
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    ASSERT_TRUE(rows <= QR_CORPUS_MAX_ROWS);
    ASSERT_TRUE(cols <= QR_CORPUS_MAX_COLS);
    if (rows > QR_CORPUS_MAX_ROWS || cols > QR_CORPUS_MAX_COLS)
        return;

    double x[QR_CORPUS_MAX_COLS] = {0.0};
    REQUIRE_OK(sparse_qr_solve_minnorm(A, b, x, NULL));

    double residual = qr_corpus_minnorm_residual(A, x, b, rows);
    double norm = vec_norm2(x, cols);
    double max_error = 0.0;
    for (idx_t col = 0; col < cols; col++) {
        double diff = fabs(x[col] - expected_x[col]);
        if (diff > max_error)
            max_error = diff;
    }
    printf("    %s minnorm residual=%.3e norm=%.15g max_solution_error=%.3e\n", fixture_key,
           residual, norm, max_error);
    ASSERT_TRUE(residual <= QR_CORPUS_MINNORM_TOL);
    ASSERT_NEAR(norm, expected_norm, QR_CORPUS_MINNORM_TOL);
    ASSERT_TRUE(max_error <= QR_CORPUS_MINNORM_TOL);
}

static SparseMatrix *qr_corpus_make_minnorm_2x4(void) {
    SparseMatrix *A = sparse_create(2, 4);
    ASSERT_NOT_NULL(A);
    if (!A)
        return NULL;
    if (!tf_qr_insert_or_free(&A, 0, 0, 1.0) || !tf_qr_insert_or_free(&A, 0, 2, 1.0) ||
        !tf_qr_insert_or_free(&A, 1, 1, 1.0) || !tf_qr_insert_or_free(&A, 1, 3, 1.0))
        return NULL;
    return A;
}

static SparseMatrix *qr_corpus_make_minnorm_3x6(void) {
    SparseMatrix *A = sparse_create(3, 6);
    ASSERT_NOT_NULL(A);
    if (!A)
        return NULL;
    if (!tf_qr_insert_or_free(&A, 0, 0, 2.0) || !tf_qr_insert_or_free(&A, 0, 3, 1.0) ||
        !tf_qr_insert_or_free(&A, 1, 1, 3.0) || !tf_qr_insert_or_free(&A, 1, 4, 1.0) ||
        !tf_qr_insert_or_free(&A, 2, 2, 1.0) || !tf_qr_insert_or_free(&A, 2, 5, 2.0))
        return NULL;
    return A;
}

static SparseMatrix *qr_corpus_make_minnorm_5x10(void) {
    SparseMatrix *A = sparse_create(5, 10);
    ASSERT_NOT_NULL(A);
    if (!A)
        return NULL;
    for (idx_t row = 0; row < 5; row++) {
        if (!tf_qr_insert_or_free(&A, row, row, 2.0) ||
            !tf_qr_insert_or_free(&A, row, row + 5, 1.0))
            return NULL;
    }
    return A;
}

static void test_qr_corpus_rankdef_6x4_fixture_shape(void) {
    SparseMatrix *A = tf_qr_make_rankdef_6x4_nullspace_v1();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    ASSERT_EQ(sparse_rows(A), 6);
    ASSERT_EQ(sparse_cols(A), 4);
    ASSERT_EQ(sparse_nnz(A), 14);

    sparse_free(A);
}

static void test_qr_corpus_rankdef_6x4_rank_and_nullity(void) {
    SparseMatrix *A = tf_qr_make_rankdef_6x4_nullspace_v1();
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

    idx_t rank = sparse_qr_rank(&qr, 0.0);
    ASSERT_EQ(rank, 3);

    idx_t nullity = -1;
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, NULL, &nullity), SPARSE_OK);
    ASSERT_EQ(nullity, 1);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_corpus_rankdef_6x4_nullspace_residual(void) {
    SparseMatrix *A = tf_qr_make_rankdef_6x4_nullspace_v1();
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

    idx_t nullity = -1;
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, NULL, &nullity), SPARSE_OK);
    ASSERT_EQ(nullity, 1);
    if (nullity != 1) {
        sparse_qr_free(&qr);
        sparse_free(A);
        return;
    }

    double basis[4] = {0.0};
    ASSERT_ERR(sparse_qr_nullspace(&qr, 0.0, basis, &nullity), SPARSE_OK);
    ASSERT_EQ(nullity, 1);

    double basis_norm = vec_norm2(basis, 4);
    ASSERT_TRUE(basis_norm > 0.0);
    double residual = tf_qr_normalized_matvec_residual(A, basis, 4);
    printf("    %s solver nullspace normalized residual = %.3e\n", QR_CORPUS_FIXTURE_KEY, residual);
    ASSERT_TRUE(residual <= QR_CORPUS_RESIDUAL_TOL);

    sparse_qr_free(&qr);
    sparse_free(A);
}

static void test_qr_corpus_rankdef_6x4_reference_direction(void) {
    SparseMatrix *A = tf_qr_make_rankdef_6x4_nullspace_v1();
    ASSERT_NOT_NULL(A);
    if (!A)
        return;

    const double reference[4] = {-1.0, -1.0, 0.0, 1.0};
    double residual = tf_qr_normalized_matvec_residual(A, reference, 4);
    printf("    %s reference direction normalized residual = %.3e\n", QR_CORPUS_FIXTURE_KEY,
           residual);
    ASSERT_TRUE(residual <= QR_CORPUS_REFERENCE_RESIDUAL_TOL);

    sparse_free(A);
}

static void test_qr_corpus_rankdef_duplicate_5x4_shape(void) {
    SparseMatrix *A = tf_qr_make_rankdef_duplicate_5x4();
    qr_corpus_assert_shape("qr_rankdef_duplicate_5x4_v1", A, 5, 4, 14);
    sparse_free(A);
}

static void test_qr_corpus_rankdef_duplicate_5x4_rank_nullity_residual_projector(void) {
    SparseMatrix *A = tf_qr_make_rankdef_duplicate_5x4();
    const double reference[4] = {0.0, -1.0, 0.0, 1.0};
    qr_corpus_assert_rankdef_fixture("qr_rankdef_duplicate_5x4_v1", A, 3, 1, reference, 4);
    sparse_free(A);
}

static void test_qr_corpus_rankdef_dependent_row_4x3_shape(void) {
    SparseMatrix *A = tf_qr_make_dependent_row_4x3();
    qr_corpus_assert_shape("qr_rankdef_dependent_row_4x3_v1", A, 4, 3, 9);
    sparse_free(A);
}

static void test_qr_corpus_rankdef_dependent_row_4x3_rank_nullity_residual_projector(void) {
    SparseMatrix *A = tf_qr_make_dependent_row_4x3();
    const double reference[3] = {-1.0, -2.0, 1.0};
    qr_corpus_assert_rankdef_fixture("qr_rankdef_dependent_row_4x3_v1", A, 2, 1, reference, 3);
    sparse_free(A);
}

static void test_qr_corpus_minnorm_2x4_shape(void) {
    SparseMatrix *A = qr_corpus_make_minnorm_2x4();
    qr_corpus_assert_shape("qr_underdetermined_minnorm_2x4", A, 2, 4, 4);
    sparse_free(A);
}

static void test_qr_corpus_minnorm_2x4_status_residual_norm_values(void) {
    SparseMatrix *A = qr_corpus_make_minnorm_2x4();
    const double b[2] = {1.0, 1.0};
    const double expected[4] = {0.5, 0.5, 0.5, 0.5};
    qr_corpus_assert_minnorm_fixture("qr_underdetermined_minnorm_2x4", A, b, expected, 2, 4, 1.0);
    sparse_free(A);
}

static void test_qr_corpus_minnorm_3x6_shape(void) {
    SparseMatrix *A = qr_corpus_make_minnorm_3x6();
    qr_corpus_assert_shape("qr_minnorm_3x6_exact_values", A, 3, 6, 6);
    sparse_free(A);
}

static void test_qr_corpus_minnorm_3x6_status_residual_norm_values(void) {
    SparseMatrix *A = qr_corpus_make_minnorm_3x6();
    const double b[3] = {3.0, 4.0, 5.0};
    const double expected[6] = {1.2, 1.2, 1.0, 0.6, 0.4, 2.0};
    qr_corpus_assert_minnorm_fixture("qr_minnorm_3x6_exact_values", A, b, expected, 3, 6,
                                     sqrt(8.4));
    sparse_free(A);
}

static void test_qr_corpus_minnorm_5x10_shape(void) {
    SparseMatrix *A = qr_corpus_make_minnorm_5x10();
    qr_corpus_assert_shape("qr_minnorm_5x10_exact_values", A, 5, 10, 10);
    sparse_free(A);
}

static void test_qr_corpus_minnorm_5x10_status_residual_norm_values(void) {
    SparseMatrix *A = qr_corpus_make_minnorm_5x10();
    const double b[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double expected[10] = {0.4, 0.8, 1.2, 1.6, 2.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    qr_corpus_assert_minnorm_fixture("qr_minnorm_5x10_exact_values", A, b, expected, 5, 10,
                                     sqrt(11.0));
    sparse_free(A);
}

int main(void) {
    TEST_SUITE_BEGIN("QR Corpus Proof");

    RUN_TEST(test_qr_corpus_rankdef_6x4_fixture_shape);
    RUN_TEST(test_qr_corpus_rankdef_6x4_rank_and_nullity);
    RUN_TEST(test_qr_corpus_rankdef_6x4_nullspace_residual);
    RUN_TEST(test_qr_corpus_rankdef_6x4_reference_direction);
    RUN_TEST(test_qr_corpus_rankdef_duplicate_5x4_shape);
    RUN_TEST(test_qr_corpus_rankdef_duplicate_5x4_rank_nullity_residual_projector);
    RUN_TEST(test_qr_corpus_rankdef_dependent_row_4x3_shape);
    RUN_TEST(test_qr_corpus_rankdef_dependent_row_4x3_rank_nullity_residual_projector);
    RUN_TEST(test_qr_corpus_minnorm_2x4_shape);
    RUN_TEST(test_qr_corpus_minnorm_2x4_status_residual_norm_values);
    RUN_TEST(test_qr_corpus_minnorm_3x6_shape);
    RUN_TEST(test_qr_corpus_minnorm_3x6_status_residual_norm_values);
    RUN_TEST(test_qr_corpus_minnorm_5x10_shape);
    RUN_TEST(test_qr_corpus_minnorm_5x10_status_residual_norm_values);

    TEST_SUITE_END();
}
