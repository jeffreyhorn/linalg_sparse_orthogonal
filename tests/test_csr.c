#include "sparse_csr.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * CSR tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Known 3x3 → verify CSR arrays */
static void test_csr_known(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    SparseCsr *csr = NULL;
    ASSERT_ERR(sparse_to_csr(A, &csr), SPARSE_OK);
    ASSERT_NOT_NULL(csr);

    ASSERT_EQ(csr->rows, 3);
    ASSERT_EQ(csr->cols, 3);
    ASSERT_EQ(csr->nnz, 5);

    /* row_ptr: [0, 2, 3, 5] */
    ASSERT_EQ(csr->row_ptr[0], 0);
    ASSERT_EQ(csr->row_ptr[1], 2);
    ASSERT_EQ(csr->row_ptr[2], 3);
    ASSERT_EQ(csr->row_ptr[3], 5);

    /* Row 0: cols 0, 2 */
    ASSERT_EQ(csr->col_idx[0], 0);
    ASSERT_NEAR(csr->values[0], 1.0, 0.0);
    ASSERT_EQ(csr->col_idx[1], 2);
    ASSERT_NEAR(csr->values[1], 3.0, 0.0);

    /* Row 1: col 1 */
    ASSERT_EQ(csr->col_idx[2], 1);
    ASSERT_NEAR(csr->values[2], 5.0, 0.0);

    /* Row 2: cols 0, 2 */
    ASSERT_EQ(csr->col_idx[3], 0);
    ASSERT_NEAR(csr->values[3], 7.0, 0.0);
    ASSERT_EQ(csr->col_idx[4], 2);
    ASSERT_NEAR(csr->values[4], 9.0, 0.0);

    sparse_csr_free(csr);
    sparse_free(A);
}

/* Round-trip: matrix → CSR → matrix → verify entries */
static void test_csr_roundtrip(void) {
    SparseMatrix *A = sparse_create(4, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 2.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 0, 4.0);
    sparse_insert(A, 2, 1, 5.0);
    sparse_insert(A, 3, 2, 6.0);

    SparseCsr *csr = NULL;
    ASSERT_ERR(sparse_to_csr(A, &csr), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_from_csr(csr, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    ASSERT_EQ(sparse_rows(B), 4);
    ASSERT_EQ(sparse_cols(B), 3);
    ASSERT_EQ(sparse_nnz(B), 6);

    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 0, 2), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 1), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 3, 2), 6.0, 0.0);

    sparse_csr_free(csr);
    sparse_free(A);
    sparse_free(B);
}

/* Empty matrix → CSR with nnz=0 */
static void test_csr_empty(void) {
    SparseMatrix *A = sparse_create(3, 3);

    SparseCsr *csr = NULL;
    ASSERT_ERR(sparse_to_csr(A, &csr), SPARSE_OK);
    ASSERT_NOT_NULL(csr);
    ASSERT_EQ(csr->nnz, 0);
    ASSERT_EQ(csr->row_ptr[0], 0);
    ASSERT_EQ(csr->row_ptr[3], 0);

    /* Round-trip */
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_from_csr(csr, &B), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(B), 0);

    sparse_csr_free(csr);
    sparse_free(A);
    sparse_free(B);
}

/* Dense 2x2 → all entries present */
static void test_csr_dense(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);

    SparseCsr *csr = NULL;
    ASSERT_ERR(sparse_to_csr(A, &csr), SPARSE_OK);
    ASSERT_EQ(csr->nnz, 4);

    /* Round-trip */
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_from_csr(csr, &B), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(B), 4);
    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 0, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 0), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 4.0, 0.0);

    sparse_csr_free(csr);
    sparse_free(A);
    sparse_free(B);
}

/* NULL inputs → proper error codes */
static void test_csr_null(void) {
    SparseCsr *csr;
    SparseMatrix *mat;
    ASSERT_ERR(sparse_to_csr(NULL, &csr), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_from_csr(NULL, &mat), SPARSE_ERR_NULL);

    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_ERR(sparse_to_csr(A, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CSC tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Known 3x3 → verify CSC arrays */
static void test_csc_known(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    SparseCsc *csc = NULL;
    ASSERT_ERR(sparse_to_csc(A, &csc), SPARSE_OK);
    ASSERT_NOT_NULL(csc);

    ASSERT_EQ(csc->nnz, 5);

    /* col_ptr: [0, 2, 3, 5] */
    ASSERT_EQ(csc->col_ptr[0], 0);
    ASSERT_EQ(csc->col_ptr[1], 2);
    ASSERT_EQ(csc->col_ptr[2], 3);
    ASSERT_EQ(csc->col_ptr[3], 5);

    /* Col 0: rows 0, 2 */
    ASSERT_EQ(csc->row_idx[0], 0);
    ASSERT_NEAR(csc->values[0], 1.0, 0.0);
    ASSERT_EQ(csc->row_idx[1], 2);
    ASSERT_NEAR(csc->values[1], 7.0, 0.0);

    sparse_csc_free(csc);
    sparse_free(A);
}

/* CSC round-trip */
static void test_csc_roundtrip(void) {
    SparseMatrix *A = sparse_create(3, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 3, 2.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 2, 3, 5.0);

    SparseCsc *csc = NULL;
    ASSERT_ERR(sparse_to_csc(A, &csc), SPARSE_OK);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_from_csc(csc, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    ASSERT_EQ(sparse_rows(B), 3);
    ASSERT_EQ(sparse_cols(B), 4);
    ASSERT_EQ(sparse_nnz(B), 5);

    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 0, 3), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 2), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 3), 5.0, 0.0);

    sparse_csc_free(csc);
    sparse_free(A);
    sparse_free(B);
}

/* CSC NULL inputs */
static void test_csc_null(void) {
    SparseCsc *csc;
    SparseMatrix *mat;
    ASSERT_ERR(sparse_to_csc(NULL, &csc), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_from_csc(NULL, &mat), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SuiteSparse validation
 * ═══════════════════════════════════════════════════════════════════════ */

/* CSR round-trip on west0067 */
static void test_csr_suitesparse_west0067(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/west0067.mtx"), SPARSE_OK);
    idx_t orig_nnz = sparse_nnz(A);

    SparseCsr *csr = NULL;
    ASSERT_ERR(sparse_to_csr(A, &csr), SPARSE_OK);
    ASSERT_EQ(csr->nnz, orig_nnz);
    ASSERT_EQ(csr->rows, 67);
    ASSERT_EQ(csr->cols, 67);

    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_from_csr(csr, &B), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(B), orig_nnz);

    /* Verify sample entries match via CSR arrays */
    for (idx_t i = 0; i < 67; i++) {
        for (idx_t k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
            ASSERT_NEAR(sparse_get_phys(B, i, csr->col_idx[k]), csr->values[k], 0.0);
        }
    }

    printf("    west0067: CSR round-trip OK, nnz=%d\n", (int)orig_nnz);

    sparse_csr_free(csr);
    sparse_free(A);
    sparse_free(B);
}

/* CSC on nos4 — verify column structure */
static void test_csc_suitesparse_nos4(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);
    idx_t orig_nnz = sparse_nnz(A);

    SparseCsc *csc = NULL;
    ASSERT_ERR(sparse_to_csc(A, &csc), SPARSE_OK);
    ASSERT_EQ(csc->nnz, orig_nnz);
    ASSERT_EQ(csc->rows, 100);
    ASSERT_EQ(csc->cols, 100);

    /* Verify col_ptr is monotonic and sums to nnz */
    for (idx_t j = 0; j < 100; j++)
        ASSERT_TRUE(csc->col_ptr[j] <= csc->col_ptr[j + 1]);
    ASSERT_EQ(csc->col_ptr[100], orig_nnz);

    /* Round-trip */
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_from_csc(csc, &B), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(B), orig_nnz);

    printf("    nos4: CSC round-trip OK, nnz=%d\n", (int)orig_nnz);

    sparse_csc_free(csc);
    sparse_free(A);
    sparse_free(B);
}

/* Transpose relationship: CSR of A should match CSC of A^T
 * Since we don't have sparse_transpose yet, build A and A^T manually */
static void test_csr_csc_transpose(void) {
    /* A = [[1, 0, 3], [0, 5, 0], [7, 0, 9]] */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    /* A^T = [[1, 0, 7], [0, 5, 0], [3, 0, 9]] */
    SparseMatrix *AT = sparse_create(3, 3);
    sparse_insert(AT, 0, 0, 1.0);
    sparse_insert(AT, 0, 2, 7.0);
    sparse_insert(AT, 1, 1, 5.0);
    sparse_insert(AT, 2, 0, 3.0);
    sparse_insert(AT, 2, 2, 9.0);

    SparseCsr *csr_A = NULL;
    ASSERT_ERR(sparse_to_csr(A, &csr_A), SPARSE_OK);

    SparseCsc *csc_AT = NULL;
    ASSERT_ERR(sparse_to_csc(AT, &csc_AT), SPARSE_OK);

    /* CSR of A and CSC of A^T should have same structure:
     * CSR row_ptr of A = CSC col_ptr of A^T
     * CSR col_idx of A = CSC row_idx of A^T
     * CSR values of A = CSC values of A^T */
    ASSERT_EQ(csr_A->nnz, csc_AT->nnz);
    for (idx_t i = 0; i <= 3; i++)
        ASSERT_EQ(csr_A->row_ptr[i], csc_AT->col_ptr[i]);
    for (idx_t k = 0; k < csr_A->nnz; k++) {
        ASSERT_EQ(csr_A->col_idx[k], csc_AT->row_idx[k]);
        ASSERT_NEAR(csr_A->values[k], csc_AT->values[k], 0.0);
    }

    sparse_csr_free(csr_A);
    sparse_csc_free(csc_AT);
    sparse_free(A);
    sparse_free(AT);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("CSR/CSC Conversion Tests");

    /* CSR */
    RUN_TEST(test_csr_known);
    RUN_TEST(test_csr_roundtrip);
    RUN_TEST(test_csr_empty);
    RUN_TEST(test_csr_dense);
    RUN_TEST(test_csr_null);

    /* CSC */
    RUN_TEST(test_csc_known);
    RUN_TEST(test_csc_roundtrip);
    RUN_TEST(test_csc_null);

    /* SuiteSparse validation */
    RUN_TEST(test_csr_suitesparse_west0067);
    RUN_TEST(test_csc_suitesparse_nos4);

    /* Transpose relationship */
    RUN_TEST(test_csr_csc_transpose);

    TEST_SUITE_END();
}
