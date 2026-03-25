#include "sparse_matrix.h"
#include "sparse_csr.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CSR tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Known 3x3 → verify CSR arrays */
static void test_csr_known(void)
{
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
static void test_csr_roundtrip(void)
{
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
static void test_csr_empty(void)
{
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
static void test_csr_dense(void)
{
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
static void test_csr_null(void)
{
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
static void test_csc_known(void)
{
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
static void test_csc_roundtrip(void)
{
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
static void test_csc_null(void)
{
    SparseCsc *csc;
    SparseMatrix *mat;
    ASSERT_ERR(sparse_to_csc(NULL, &csc), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_from_csc(NULL, &mat), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
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

    TEST_SUITE_END();
}
