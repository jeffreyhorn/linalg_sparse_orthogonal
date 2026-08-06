#include "sparse_matrix.h"
#include "sparse_qr.h"
#include "sparse_types.h"
#include "test_framework.h"
#include "test_qr_helpers.h"

#include <math.h>

#define QR_CORPUS_FIXTURE_KEY "qr_rank_deficient_6x4_nullspace_v1"
#define QR_CORPUS_RESIDUAL_TOL 1e-10
#define QR_CORPUS_REFERENCE_RESIDUAL_TOL 1e-12

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

int main(void) {
    TEST_SUITE_BEGIN("QR Corpus Proof");

    RUN_TEST(test_qr_corpus_rankdef_6x4_fixture_shape);
    RUN_TEST(test_qr_corpus_rankdef_6x4_rank_and_nullity);
    RUN_TEST(test_qr_corpus_rankdef_6x4_nullspace_residual);
    RUN_TEST(test_qr_corpus_rankdef_6x4_reference_direction);

    TEST_SUITE_END();
}
