#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

/*
 * Path helper: tests must find data files relative to the executable.
 * We support both running from the project root (make test) and from
 * the build directory. The DATA_DIR macro can be overridden at compile
 * time; if not set, we default to "tests/data".
 */
#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Save / Load round-trip tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_roundtrip_basic(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    ASSERT_ERR(sparse_save_mm(A, "/tmp/test_rt_basic.mtx"), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/test_rt_basic.mtx"), SPARSE_OK);
    ASSERT_NOT_NULL(B);
    ASSERT_EQ(sparse_rows(B), 3);
    ASSERT_EQ(sparse_cols(B), 3);
    ASSERT_EQ(sparse_nnz(B), 5);

    /* Compare all elements */
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++)
            ASSERT_NEAR(sparse_get_phys(B, i, j), sparse_get_phys(A, i, j), 1e-14);

    sparse_free(A);
    sparse_free(B);
}

static void test_roundtrip_single_element(void) {
    SparseMatrix *A = sparse_create(5, 5);
    sparse_insert(A, 2, 3, 42.0);
    ASSERT_EQ(sparse_nnz(A), 1);

    ASSERT_ERR(sparse_save_mm(A, "/tmp/test_rt_single.mtx"), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/test_rt_single.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(B), 1);
    ASSERT_NEAR(sparse_get_phys(B, 2, 3), 42.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

static void test_roundtrip_rectangular(void) {
    SparseMatrix *A = sparse_create(3, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 4, 3.0);

    ASSERT_ERR(sparse_save_mm(A, "/tmp/test_rt_rect.mtx"), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/test_rt_rect.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_rows(B), 3);
    ASSERT_EQ(sparse_cols(B), 5);
    ASSERT_EQ(sparse_nnz(B), 3);
    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 2), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 4), 3.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

static void test_roundtrip_precision(void) {
    /* Test that extreme values survive the round-trip */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1e300);
    sparse_insert(A, 1, 1, 1e-300);
    sparse_insert(A, 2, 2, -3.141592653589793);

    ASSERT_ERR(sparse_save_mm(A, "/tmp/test_rt_prec.mtx"), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/test_rt_prec.mtx"), SPARSE_OK);

    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 1e300, 1e285);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 1e-300, 1e-314);
    ASSERT_NEAR(sparse_get_phys(B, 2, 2), -3.141592653589793, 1e-14);

    sparse_free(A);
    sparse_free(B);
}

static void test_roundtrip_nnz_preserved(void) {
    /* Dense-ish 4x4 */
    SparseMatrix *A = sparse_create(4, 4);
    for (idx_t i = 0; i < 4; i++)
        for (idx_t j = 0; j < 4; j++)
            sparse_insert(A, i, j, (double)(i * 4 + j + 1));

    idx_t nnz_orig = sparse_nnz(A);
    ASSERT_EQ(nnz_orig, 16);

    ASSERT_ERR(sparse_save_mm(A, "/tmp/test_rt_nnz.mtx"), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/test_rt_nnz.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(B), nnz_orig);

    sparse_free(A);
    sparse_free(B);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Loading test data files
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_load_identity(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, DATA_DIR "/identity_5.mtx"), SPARSE_OK);
    ASSERT_NOT_NULL(A);
    ASSERT_EQ(sparse_rows(A), 5);
    ASSERT_EQ(sparse_cols(A), 5);
    ASSERT_EQ(sparse_nnz(A), 5);

    for (idx_t i = 0; i < 5; i++)
        for (idx_t j = 0; j < 5; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(sparse_get_phys(A, i, j), expected, 0.0);
        }

    sparse_free(A);
}

static void test_load_diagonal(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, DATA_DIR "/diagonal_10.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_rows(A), 10);
    ASSERT_EQ(sparse_nnz(A), 10);

    for (idx_t i = 0; i < 10; i++)
        ASSERT_NEAR(sparse_get_phys(A, i, i), (double)(i + 1), 0.0);

    sparse_free(A);
}

static void test_load_symmetric(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, DATA_DIR "/symmetric_4.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_rows(A), 4);
    ASSERT_EQ(sparse_cols(A), 4);

    /* Verify symmetry: A(i,j) == A(j,i) */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 10.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 1), 1.0, 0.0); /* mirrored */
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 12.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 2), 2.0, 0.0); /* mirrored */
    ASSERT_NEAR(sparse_get_phys(A, 2, 1), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 2), 1.0, 0.0); /* mirrored */
    ASSERT_NEAR(sparse_get_phys(A, 2, 2), 14.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 3, 3), 16.0, 0.0);
    /* Off-diag zeros */
    ASSERT_NEAR(sparse_get_phys(A, 3, 0), 0.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 3), 0.0, 0.0);

    sparse_free(A);
}

static void test_load_pattern(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, DATA_DIR "/pattern_3.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_rows(A), 3);
    ASSERT_EQ(sparse_nnz(A), 5);

    /* Pattern matrices should have value 1.0 */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 2), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 2), 1.0, 0.0);
    /* Zeros */
    ASSERT_NEAR(sparse_get_phys(A, 0, 1), 0.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 0), 0.0, 0.0);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Error path tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_load_nonexistent_file(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, "/tmp/no_such_file_xyz.mtx"), SPARSE_ERR_IO);
    ASSERT_NULL(A);
}

static void test_load_bad_header(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, DATA_DIR "/bad_header.mtx"), SPARSE_ERR_PARSE);
    ASSERT_NULL(A);
}

static void test_save_null_args(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_ERR(sparse_save_mm(NULL, "/tmp/test.mtx"), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_save_mm(A, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_load_null_args(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(NULL, "test.mtx"), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_load_mm(&A, NULL), SPARSE_ERR_NULL);
}

static void test_save_invalid_path(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    ASSERT_ERR(sparse_save_mm(A, "/no_such_dir/test.mtx"), SPARSE_ERR_IO);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * errno capture tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_load_errno_enoent(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, "/tmp/no_such_file_errno_test.mtx");
    ASSERT_ERR(err, SPARSE_ERR_IO);
    ASSERT_NULL(A);
    ASSERT_EQ(sparse_errno(), ENOENT);
}

static void test_save_errno_bad_path(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_err_t err = sparse_save_mm(A, "/no_such_dir/errno_test.mtx");
    ASSERT_ERR(err, SPARSE_ERR_IO);
    ASSERT_TRUE(sparse_errno() != 0);
    sparse_free(A);
}

static void test_errno_cleared_on_success(void) {
    /* First, trigger an I/O error to set sparse_errno */
    SparseMatrix *A = NULL;
    sparse_load_mm(&A, "/tmp/no_such_file_clear_test.mtx");
    ASSERT_TRUE(sparse_errno() != 0);

    /* Now do a successful I/O operation */
    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 1.0);
    ASSERT_ERR(sparse_save_mm(B, "/tmp/test_errno_clear.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_errno(), 0);

    /* Also verify load clears it */
    sparse_load_mm(&A, "/tmp/no_such_file_clear_test2.mtx");
    ASSERT_TRUE(sparse_errno() != 0);
    ASSERT_ERR(sparse_load_mm(&A, "/tmp/test_errno_clear.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_errno(), 0);

    sparse_free(A);
    sparse_free(B);
}

static void test_strerror_io(void) {
    const char *msg = sparse_strerror(SPARSE_ERR_IO);
    ASSERT_NOT_NULL(msg);
    ASSERT_TRUE(strlen(msg) > 0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Format edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_roundtrip_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 99.5);

    ASSERT_ERR(sparse_save_mm(A, "/tmp/test_rt_1x1.mtx"), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/test_rt_1x1.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_rows(B), 1);
    ASSERT_EQ(sparse_cols(B), 1);
    ASSERT_EQ(sparse_nnz(B), 1);
    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 99.5, 0.0);

    sparse_free(A);
    sparse_free(B);
}

static void test_roundtrip_empty(void) {
    SparseMatrix *A = sparse_create(5, 5);
    /* nnz == 0 */
    ASSERT_ERR(sparse_save_mm(A, "/tmp/test_rt_empty.mtx"), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/test_rt_empty.mtx"), SPARSE_OK);
    ASSERT_EQ(sparse_rows(B), 5);
    ASSERT_EQ(sparse_cols(B), 5);
    ASSERT_EQ(sparse_nnz(B), 0);

    sparse_free(A);
    sparse_free(B);
}

static void test_roundtrip_negative_values(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, -1.0);
    sparse_insert(A, 0, 1, -2.5);
    sparse_insert(A, 1, 0, -3.0);
    sparse_insert(A, 1, 1, -4.0);

    ASSERT_ERR(sparse_save_mm(A, "/tmp/test_rt_neg.mtx"), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/test_rt_neg.mtx"), SPARSE_OK);
    ASSERT_NEAR(sparse_get_phys(B, 0, 0), -1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 0, 1), -2.5, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 0), -3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), -4.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

static void test_roundtrip_after_permutation(void) {
    /*
     * Save a matrix that has been factored (permutations scrambled).
     * sparse_save_mm writes in logical order, so the round-trip should
     * produce the logically-equivalent matrix.
     */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    /* Save the original values in logical order for comparison */
    double orig[3][3];
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++)
            orig[i][j] = sparse_get(A, i, j);

    /* Factor to scramble permutations */
    sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12);
    /* Note: A now contains L+U in permuted form */

    ASSERT_ERR(sparse_save_mm(A, "/tmp/test_rt_perm.mtx"), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_load_mm(&B, "/tmp/test_rt_perm.mtx"), SPARSE_OK);

    /* B should have identity perms but same logical layout as A */
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++)
            ASSERT_NEAR(sparse_get_phys(B, i, j), sparse_get(A, i, j), 1e-14);

    sparse_free(A);
    sparse_free(B);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sparse Matrix I/O Tests");

    /* Round-trip */
    RUN_TEST(test_roundtrip_basic);
    RUN_TEST(test_roundtrip_single_element);
    RUN_TEST(test_roundtrip_rectangular);
    RUN_TEST(test_roundtrip_precision);
    RUN_TEST(test_roundtrip_nnz_preserved);

    /* Loading test data */
    RUN_TEST(test_load_identity);
    RUN_TEST(test_load_diagonal);
    RUN_TEST(test_load_symmetric);
    RUN_TEST(test_load_pattern);

    /* Error paths */
    RUN_TEST(test_load_nonexistent_file);
    RUN_TEST(test_load_bad_header);
    RUN_TEST(test_save_null_args);
    RUN_TEST(test_load_null_args);
    RUN_TEST(test_save_invalid_path);

    /* errno capture */
    RUN_TEST(test_load_errno_enoent);
    RUN_TEST(test_save_errno_bad_path);
    RUN_TEST(test_errno_cleared_on_success);
    RUN_TEST(test_strerror_io);

    /* Edge cases */
    RUN_TEST(test_roundtrip_1x1);
    RUN_TEST(test_roundtrip_empty);
    RUN_TEST(test_roundtrip_negative_values);
    RUN_TEST(test_roundtrip_after_permutation);

    TEST_SUITE_END();
}
