#include "sparse_lu.h"
#include "sparse_lu_csr.h"
#include "sparse_matrix.h"
#include "sparse_matrix_internal.h" /* for direct struct access in permutation test */
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
 * Helper: compare two SparseMatrices entry-by-entry (identity perms)
 * ═══════════════════════════════════════════════════════════════════════ */

static void assert_matrices_equal(const SparseMatrix *A, const SparseMatrix *B, double tol) {
    ASSERT_EQ(A->rows, B->rows);
    ASSERT_EQ(A->cols, B->cols);
    for (idx_t i = 0; i < A->rows; i++) {
        for (idx_t j = 0; j < A->cols; j++) {
            double a = sparse_get(A, i, j);
            double b = sparse_get(B, i, j);
            if (fabs(a - b) > tol) {
                TF_FAIL_("Entry (%d,%d): %.15g vs %.15g, diff=%.3e > tol=%.3e", (int)i, (int)j, a,
                         b, fabs(a - b), tol);
            }
            tf_asserts++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: NULL / error argument handling
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_csr_null_args(void) {
    LuCsr *csr = NULL;
    SparseMatrix *mat = NULL;

    ASSERT_ERR(lu_csr_from_sparse(NULL, 2.0, &csr), SPARSE_ERR_NULL);
    ASSERT_NULL(csr);

    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, NULL), SPARSE_ERR_NULL);
    sparse_free(A);

    ASSERT_ERR(lu_csr_to_sparse(NULL, &mat), SPARSE_ERR_NULL);
    ASSERT_NULL(mat);

    /* Non-square matrix should fail */
    SparseMatrix *rect = sparse_create(3, 5);
    ASSERT_ERR(lu_csr_from_sparse(rect, 2.0, &csr), SPARSE_ERR_SHAPE);
    ASSERT_NULL(csr);
    sparse_free(rect);

    /* Free NULL should be safe */
    lu_csr_free(NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Identity matrix round-trip
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_csr_identity(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);
    ASSERT_NOT_NULL(csr);

    /* Verify CSR structure */
    ASSERT_EQ(csr->n, n);
    ASSERT_EQ(csr->nnz, n);
    ASSERT_TRUE(csr->capacity >= n);

    /* Each row should have exactly 1 entry on the diagonal */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_EQ(csr->row_ptr[i + 1] - csr->row_ptr[i], 1);
        ASSERT_EQ(csr->col_idx[csr->row_ptr[i]], i);
        ASSERT_NEAR(csr->values[csr->row_ptr[i]], 1.0, 1e-15);
    }

    /* Round-trip back to SparseMatrix */
    SparseMatrix *B = NULL;
    ASSERT_ERR(lu_csr_to_sparse(csr, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    assert_matrices_equal(A, B, 1e-15);

    sparse_free(A);
    sparse_free(B);
    lu_csr_free(csr);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Dense 5×5 matrix round-trip
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_csr_dense_5x5(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);

    /* Fill with known values: A[i][j] = (i+1)*10 + (j+1) */
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            sparse_insert(A, i, j, (double)((i + 1) * 10 + (j + 1)));

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 1.5, &csr), SPARSE_OK);
    ASSERT_NOT_NULL(csr);

    ASSERT_EQ(csr->n, n);
    ASSERT_EQ(csr->nnz, n * n);

    /* Verify all entries are present and in row-major column-sorted order */
    for (idx_t i = 0; i < n; i++) {
        idx_t row_nnz = csr->row_ptr[i + 1] - csr->row_ptr[i];
        ASSERT_EQ(row_nnz, n);
        for (idx_t k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
            idx_t j = csr->col_idx[k];
            double expected = (double)((i + 1) * 10 + (j + 1));
            ASSERT_NEAR(csr->values[k], expected, 1e-15);
        }
        /* Verify column indices are sorted */
        for (idx_t k = csr->row_ptr[i] + 1; k < csr->row_ptr[i + 1]; k++) {
            ASSERT_TRUE(csr->col_idx[k] > csr->col_idx[k - 1]);
        }
    }

    /* Round-trip */
    SparseMatrix *B = NULL;
    ASSERT_ERR(lu_csr_to_sparse(csr, &B), SPARSE_OK);
    assert_matrices_equal(A, B, 1e-15);

    sparse_free(A);
    sparse_free(B);
    lu_csr_free(csr);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Empty matrix (0 nonzeros)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_csr_empty(void) {
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);
    ASSERT_NOT_NULL(csr);
    ASSERT_EQ(csr->n, n);
    ASSERT_EQ(csr->nnz, 0);

    /* All row pointers should be 0 */
    for (idx_t i = 0; i <= n; i++)
        ASSERT_EQ(csr->row_ptr[i], 0);

    /* Round-trip: should produce empty matrix */
    SparseMatrix *B = NULL;
    ASSERT_ERR(lu_csr_to_sparse(csr, &B), SPARSE_OK);
    ASSERT_EQ(B->nnz, 0);

    sparse_free(A);
    sparse_free(B);
    lu_csr_free(csr);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: 1×1 matrix
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_csr_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 42.0);

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);
    ASSERT_EQ(csr->n, 1);
    ASSERT_EQ(csr->nnz, 1);
    ASSERT_EQ(csr->col_idx[0], 0);
    ASSERT_NEAR(csr->values[0], 42.0, 1e-15);

    SparseMatrix *B = NULL;
    ASSERT_ERR(lu_csr_to_sparse(csr, &B), SPARSE_OK);
    assert_matrices_equal(A, B, 1e-15);

    sparse_free(A);
    sparse_free(B);
    lu_csr_free(csr);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Permuted matrix — verify logical ordering is correct
 * After LU factorization, row_perm/col_perm are non-identity.
 * Convert to LuCsr, convert back, and verify entries match the
 * original (pre-factored) logical view.
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_csr_with_permutations(void) {
    /*
     * Build a 4×4 matrix and manually set non-identity permutations.
     * A (logical view):
     *   [4 1 0 0]
     *   [1 4 1 0]
     *   [0 1 4 1]
     *   [0 0 1 4]
     *
     * We'll swap rows 0↔2 and cols 1↔3 in the permutation arrays
     * (without moving physical data) to simulate a pivoted state.
     */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);

    /* Insert in physical order (identity perm initially) */
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

    /* Snapshot the original logical view before permuting */
    double orig[4][4];
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            orig[i][j] = sparse_get(A, i, j);

    /* Manually swap row perm: logical 0 ↔ physical 2, logical 2 ↔ physical 0 */
    A->row_perm[0] = 2;
    A->row_perm[2] = 0;
    A->inv_row_perm[0] = 2;
    A->inv_row_perm[2] = 0;

    /* Swap col perm: logical 1 ↔ physical 3, logical 3 ↔ physical 1 */
    A->col_perm[1] = 3;
    A->col_perm[3] = 1;
    A->inv_col_perm[1] = 3;
    A->inv_col_perm[3] = 1;

    /* After permutation, the logical view is different.
     * Capture the new logical view. */
    double permuted[4][4];
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            permuted[i][j] = sparse_get(A, i, j);

    /* Convert to LuCsr — should reflect the permuted logical view */
    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);
    ASSERT_NOT_NULL(csr);

    /* Verify CSR matches the permuted logical view */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; k++) {
            idx_t j = csr->col_idx[k];
            ASSERT_NEAR(csr->values[k], permuted[i][j], 1e-15);
        }
    }

    /* Round-trip: LuCsr → SparseMatrix should give the permuted logical view
     * with identity permutations */
    SparseMatrix *B = NULL;
    ASSERT_ERR(lu_csr_to_sparse(csr, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            ASSERT_NEAR(sparse_get(B, i, j), permuted[i][j], 1e-15);

    sparse_free(A);
    sparse_free(B);
    lu_csr_free(csr);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Fill factor clamping
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_csr_fill_factor_clamping(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);

    /* fill_factor < 1.0 should be clamped to 1.0 */
    LuCsr *csr1 = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 0.5, &csr1), SPARSE_OK);
    ASSERT_TRUE(csr1->capacity >= csr1->nnz);
    lu_csr_free(csr1);

    /* fill_factor > 20.0 should be clamped to 20.0 */
    LuCsr *csr2 = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 100.0, &csr2), SPARSE_OK);
    ASSERT_TRUE(csr2->capacity <= 20 * csr2->nnz + 1);
    lu_csr_free(csr2);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: SuiteSparse orsirr_1 round-trip (if available)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_csr_suitesparse_orsirr1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/orsirr_1.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] orsirr_1.mtx not available\n");
        return;
    }

    /* orsirr_1 is 1030×1030 */
    ASSERT_EQ(A->rows, 1030);
    ASSERT_EQ(A->cols, 1030);

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);
    ASSERT_NOT_NULL(csr);
    ASSERT_EQ(csr->n, 1030);
    ASSERT_EQ(csr->nnz, A->nnz);

    /* Round-trip and verify */
    SparseMatrix *B = NULL;
    ASSERT_ERR(lu_csr_to_sparse(csr, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);
    ASSERT_EQ(B->nnz, A->nnz);

    /* Spot-check: verify all entries match */
    assert_matrices_equal(A, B, 1e-15);

    sparse_free(A);
    sparse_free(B);
    lu_csr_free(csr);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test: Sparse tridiagonal matrix round-trip
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_csr_tridiagonal(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);

    /* Tridiagonal: diag=4, off-diag=-1 */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.0);
    }

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);

    /* Verify nnz = n + 2*(n-1) = 3n - 2 */
    ASSERT_EQ(csr->nnz, 3 * n - 2);

    /* Verify column ordering within each row */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t k = csr->row_ptr[i] + 1; k < csr->row_ptr[i + 1]; k++) {
            ASSERT_TRUE(csr->col_idx[k] > csr->col_idx[k - 1]);
        }
    }

    /* Round-trip */
    SparseMatrix *B = NULL;
    ASSERT_ERR(lu_csr_to_sparse(csr, &B), SPARSE_OK);
    assert_matrices_equal(A, B, 1e-15);

    sparse_free(A);
    sparse_free(B);
    lu_csr_free(csr);
}

/* ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("LU CSR Working Format");

    RUN_TEST(test_lu_csr_null_args);
    RUN_TEST(test_lu_csr_identity);
    RUN_TEST(test_lu_csr_dense_5x5);
    RUN_TEST(test_lu_csr_empty);
    RUN_TEST(test_lu_csr_1x1);
    RUN_TEST(test_lu_csr_with_permutations);
    RUN_TEST(test_lu_csr_fill_factor_clamping);
    RUN_TEST(test_lu_csr_suitesparse_orsirr1);
    RUN_TEST(test_lu_csr_tridiagonal);

    TEST_SUITE_END();
}
