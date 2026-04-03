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

/* ═══════════════════════════════════════════════════════════════════════
 * Elimination tests
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Helper: extract L and U from a factored LuCsr and verify P*A = L*U.
 * L is unit lower triangular (diag=1, below diag from CSR).
 * U is upper triangular (diag and above from CSR).
 * piv_perm[k] = original row that ended up in position k.
 */
static void verify_lu_factorization(const SparseMatrix *A_orig, const LuCsr *lu, const idx_t *piv,
                                    double tol_check) {
    idx_t n = lu->n;

    /* Build dense L, U from factored CSR */
    double *L = calloc((size_t)n * (size_t)n, sizeof(double));
    double *U = calloc((size_t)n * (size_t)n, sizeof(double));
    ASSERT_NOT_NULL(L);
    ASSERT_NOT_NULL(U);

    /* Unit diagonal for L */
    for (idx_t i = 0; i < n; i++)
        L[i * n + i] = 1.0;

    for (idx_t i = 0; i < n; i++) {
        for (idx_t p = lu->row_ptr[i]; p < lu->row_ptr[i + 1]; p++) {
            idx_t j = lu->col_idx[p];
            double v = lu->values[p];
            if (j < i)
                L[i * n + j] = v; /* L entry (below diagonal) */
            else
                U[i * n + j] = v; /* U entry (diagonal and above) */
        }
    }

    /* Compute L*U (dense) */
    double *LU = calloc((size_t)n * (size_t)n, sizeof(double));
    ASSERT_NOT_NULL(LU);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            for (idx_t k = 0; k < n; k++)
                LU[i * n + j] += L[i * n + k] * U[k * n + j];

    /* Compare L*U with P*A (permuted rows of A) */
    for (idx_t i = 0; i < n; i++) {
        idx_t orig_row = piv[i];
        for (idx_t j = 0; j < n; j++) {
            double a_val = sparse_get(A_orig, orig_row, j);
            double lu_val = LU[i * n + j];
            if (fabs(a_val - lu_val) > tol_check) {
                TF_FAIL_("P*A vs L*U mismatch at (%d,%d): P*A=%.15g, L*U=%.15g, diff=%.3e", (int)i,
                         (int)j, a_val, lu_val, fabs(a_val - lu_val));
            }
            tf_asserts++;
        }
    }

    free(L);
    free(U);
    free(LU);
}

/* Test: 3×3 known LU factorization */
static void test_lu_csr_eliminate_3x3(void) {
    /*
     * A = [2 1 1]
     *     [4 3 3]
     *     [8 7 9]
     */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 0, 4.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 3.0);
    sparse_insert(A, 2, 0, 8.0);
    sparse_insert(A, 2, 1, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);

    idx_t piv[3];
    ASSERT_ERR(lu_csr_eliminate(csr, 1e-12, 1e-14, piv), SPARSE_OK);

    verify_lu_factorization(A, csr, piv, 1e-12);

    sparse_free(A);
    lu_csr_free(csr);
}

/* Test: 5×5 dense matrix — compare CSR LU solve with linked-list LU solve */
static void test_lu_csr_eliminate_5x5_solve(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);

    /* Well-conditioned 5×5 matrix */
    double vals[5][5] = {
        {10, 1, 2, 0, 1}, {1, 8, 1, 3, 0}, {2, 1, 12, 1, 2}, {0, 3, 1, 9, 1}, {1, 0, 2, 1, 7}};
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            if (vals[i][j] != 0.0)
                sparse_insert(A, i, j, vals[i][j]);

    /* Solve with linked-list LU */
    double b[5] = {14.0, 13.0, 18.0, 14.0, 11.0};
    double x_ll[5];
    {
        SparseMatrix *Acopy = sparse_copy(A);
        ASSERT_ERR(sparse_lu_factor(Acopy, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
        ASSERT_ERR(sparse_lu_solve(Acopy, b, x_ll), SPARSE_OK);
        sparse_free(Acopy);
    }

    /* Solve with CSR LU: factor, then manually do forward/backward substitution */
    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);

    idx_t piv[5];
    ASSERT_ERR(lu_csr_eliminate(csr, 1e-12, 1e-14, piv), SPARSE_OK);

    /* Verify P*A = L*U */
    verify_lu_factorization(A, csr, piv, 1e-10);

    /* Manual forward/backward substitution using the CSR L\U factors.
     * pb = P*b */
    double pb[5];
    for (idx_t i = 0; i < n; i++)
        pb[i] = b[piv[i]];

    /* Forward substitution: L*y = pb (L has unit diagonal) */
    double y[5];
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (idx_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; p++) {
            idx_t j = csr->col_idx[p];
            if (j < i)
                sum += csr->values[p] * y[j];
        }
        y[i] = pb[i] - sum;
    }

    /* Backward substitution: U*x = y */
    double x_csr[5];
    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        double u_ii = 0.0;
        for (idx_t p = csr->row_ptr[i]; p < csr->row_ptr[i + 1]; p++) {
            idx_t j = csr->col_idx[p];
            if (j == i)
                u_ii = csr->values[p];
            else if (j > i)
                sum += csr->values[p] * x_csr[j];
        }
        ASSERT_TRUE(fabs(u_ii) > 1e-14);
        x_csr[i] = (y[i] - sum) / u_ii;
    }

    /* Verify CSR solution matches linked-list solution */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_csr[i], x_ll[i], 1e-10);

    /* Verify residual: ||A*x - b|| < tol */
    double r[5] = {0};
    sparse_matvec(A, x_csr, r);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(r[i], b[i], 1e-10);

    sparse_free(A);
    lu_csr_free(csr);
}

/* Test: Diagonal matrix — trivial factorization */
static void test_lu_csr_eliminate_diagonal(void) {
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1) * 3.0);

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);

    idx_t piv[6];
    ASSERT_ERR(lu_csr_eliminate(csr, 1e-12, 1e-14, piv), SPARSE_OK);

    /* L should be identity, U should equal A.
     * So each row i should have exactly one entry: (i, val) */
    for (idx_t i = 0; i < n; i++) {
        idx_t row_nnz = csr->row_ptr[i + 1] - csr->row_ptr[i];
        ASSERT_EQ(row_nnz, 1);
        ASSERT_EQ(csr->col_idx[csr->row_ptr[i]], i);
        ASSERT_NEAR(csr->values[csr->row_ptr[i]], (double)(i + 1) * 3.0, 1e-14);
    }

    /* No pivoting should have occurred (diagonal is already maximal) */
    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(piv[i], i);

    verify_lu_factorization(A, csr, piv, 1e-14);

    sparse_free(A);
    lu_csr_free(csr);
}

/* Test: Tridiagonal matrix — tests fill-in behavior */
static void test_lu_csr_eliminate_tridiag(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.0);
    }

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 3.0, &csr), SPARSE_OK);

    idx_t piv[10];
    ASSERT_ERR(lu_csr_eliminate(csr, 1e-12, 1e-14, piv), SPARSE_OK);

    verify_lu_factorization(A, csr, piv, 1e-10);

    /* Tridiagonal with diag-dominant: no pivoting needed */
    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(piv[i], i);

    sparse_free(A);
    lu_csr_free(csr);
}

/* Test: Matrix requiring pivoting */
static void test_lu_csr_eliminate_needs_pivot(void) {
    /*
     * A = [0 1]  — requires row swap
     *     [1 0]
     */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);

    idx_t piv[2];
    ASSERT_ERR(lu_csr_eliminate(csr, 1e-12, 1e-14, piv), SPARSE_OK);

    /* Row 1 should have been swapped to position 0 */
    ASSERT_EQ(piv[0], 1);
    ASSERT_EQ(piv[1], 0);

    verify_lu_factorization(A, csr, piv, 1e-14);

    sparse_free(A);
    lu_csr_free(csr);
}

/* Test: Singular matrix — should return SPARSE_ERR_SINGULAR */
static void test_lu_csr_eliminate_singular(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 4.0); /* row 1 = 2 * row 0 */
    sparse_insert(A, 2, 2, 1.0);

    LuCsr *csr = NULL;
    ASSERT_ERR(lu_csr_from_sparse(A, 2.0, &csr), SPARSE_OK);

    idx_t piv[3];
    ASSERT_ERR(lu_csr_eliminate(csr, 1e-12, 1e-14, piv), SPARSE_ERR_SINGULAR);

    sparse_free(A);
    lu_csr_free(csr);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 3: Integration tests — lu_csr_solve / lu_csr_factor_solve
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: compute ||A*x - b||_inf */
static double residual_norminf(const SparseMatrix *A, const double *x, const double *b, idx_t n) {
    double *r = malloc((size_t)n * sizeof(double));
    if (!r)
        return -1.0;
    sparse_matvec(A, x, r);
    double mx = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(r[i] - b[i]);
        if (d > mx)
            mx = d;
    }
    free(r);
    return mx;
}

/* Test: lu_csr_solve null args */
static void test_lu_csr_solve_null(void) {
    LuCsr csr = {0};
    idx_t piv = 0;
    double b = 0, x = 0;
    ASSERT_ERR(lu_csr_solve(NULL, &piv, &b, &x), SPARSE_ERR_NULL);
    ASSERT_ERR(lu_csr_solve(&csr, NULL, &b, &x), SPARSE_ERR_NULL);
    ASSERT_ERR(lu_csr_solve(&csr, &piv, NULL, &x), SPARSE_ERR_NULL);
    ASSERT_ERR(lu_csr_solve(&csr, &piv, &b, NULL), SPARSE_ERR_NULL);
}

/* Test: lu_csr_factor_solve on 10×10 tridiagonal */
static void test_lu_csr_factor_solve_tridiag(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.0);
    }

    /* RHS: b[i] = i + 1 */
    double b[10], x_csr[10], x_ll[10];
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    /* CSR path */
    ASSERT_ERR(lu_csr_factor_solve(A, b, x_csr, 1e-12), SPARSE_OK);

    /* Linked-list path */
    SparseMatrix *Acopy = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(Acopy, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(Acopy, b, x_ll), SPARSE_OK);
    sparse_free(Acopy);

    /* Solutions should match */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_csr[i], x_ll[i], 1e-10);

    /* Residual should be small */
    double res = residual_norminf(A, x_csr, b, n);
    ASSERT_TRUE(res < 1e-10);

    sparse_free(A);
}

/* Test: lu_csr_factor_solve on dense 8×8 */
static void test_lu_csr_factor_solve_dense(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);

    /* Diagonally dominant dense matrix */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            double v = (i == j) ? (double)(n + 1) : 1.0;
            sparse_insert(A, i, j, v);
        }
    }

    double b[8], x_csr[8], x_ll[8];
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i * i + 1);

    ASSERT_ERR(lu_csr_factor_solve(A, b, x_csr, 1e-12), SPARSE_OK);

    SparseMatrix *Acopy = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(Acopy, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(Acopy, b, x_ll), SPARSE_OK);
    sparse_free(Acopy);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_csr[i], x_ll[i], 1e-10);

    double res = residual_norminf(A, x_csr, b, n);
    ASSERT_TRUE(res < 1e-10);

    sparse_free(A);
}

/* Test: SuiteSparse orsirr_1 — solve and compare residuals */
static void test_lu_csr_factor_solve_orsirr1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/orsirr_1.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] orsirr_1.mtx not available\n");
        return;
    }

    idx_t n = sparse_rows(A);
    ASSERT_EQ(n, 1030);

    /* RHS: b[i] = sin(i) */
    double *b = malloc((size_t)n * sizeof(double));
    double *x_csr = malloc((size_t)n * sizeof(double));
    double *x_ll = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x_csr);
    ASSERT_NOT_NULL(x_ll);

    for (idx_t i = 0; i < n; i++)
        b[i] = sin((double)i);

    /* CSR path */
    ASSERT_ERR(lu_csr_factor_solve(A, b, x_csr, 1e-12), SPARSE_OK);

    /* Linked-list path */
    SparseMatrix *Acopy = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(Acopy, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(Acopy, b, x_ll), SPARSE_OK);
    sparse_free(Acopy);

    /* Both residuals should be small (solutions may differ due to different
     * pivoting order, but residuals should both be near machine precision) */
    double res_csr = residual_norminf(A, x_csr, b, n);
    double res_ll = residual_norminf(A, x_ll, b, n);
    printf("    orsirr_1 residuals: CSR=%.3e  LL=%.3e\n", res_csr, res_ll);
    ASSERT_TRUE(res_csr < 1e-6);
    ASSERT_TRUE(res_ll < 1e-6);

    free(b);
    free(x_csr);
    free(x_ll);
    sparse_free(A);
}

/* Test: SuiteSparse steam1 — solve and compare */
static void test_lu_csr_factor_solve_steam1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/steam1.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] steam1.mtx not available\n");
        return;
    }

    idx_t n = sparse_rows(A);
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x);

    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    ASSERT_ERR(lu_csr_factor_solve(A, b, x, 1e-12), SPARSE_OK);

    double res = residual_norminf(A, x, b, n);
    printf("    steam1 residual: %.3e\n", res);
    ASSERT_TRUE(res < 1e-6);

    free(b);
    free(x);
    sparse_free(A);
}

/* Test: Benchmark CSR vs linked-list on orsirr_1 */
static void test_lu_csr_benchmark_orsirr1(void) {
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/orsirr_1.mtx");
    if (err != SPARSE_OK) {
        printf("    [SKIP] orsirr_1.mtx not available\n");
        return;
    }

    idx_t n = sparse_rows(A);
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(x);

    for (idx_t i = 0; i < n; i++)
        b[i] = sin((double)i);

    /* Benchmark linked-list LU */
    clock_t t0 = clock();
    for (int trial = 0; trial < 3; trial++) {
        SparseMatrix *Acopy = sparse_copy(A);
        err = sparse_lu_factor(Acopy, SPARSE_PIVOT_PARTIAL, 1e-12);
        if (err == SPARSE_OK)
            sparse_lu_solve(Acopy, b, x);
        sparse_free(Acopy);
    }
    clock_t t1 = clock();
    double ll_time = (double)(t1 - t0) / CLOCKS_PER_SEC / 3.0;

    /* Benchmark CSR LU */
    clock_t t2 = clock();
    for (int trial = 0; trial < 3; trial++) {
        lu_csr_factor_solve(A, b, x, 1e-12);
    }
    clock_t t3 = clock();
    double csr_time = (double)(t3 - t2) / CLOCKS_PER_SEC / 3.0;

    double speedup = (csr_time > 0.0) ? ll_time / csr_time : 0.0;
    printf("    orsirr_1 (n=%d): LL=%.4fs  CSR=%.4fs  speedup=%.2fx\n", (int)n, ll_time, csr_time,
           speedup);

    /* We expect CSR to be faster. Don't hard-assert speedup since
     * it depends on hardware, but log it for inspection. */
    ASSERT_TRUE(csr_time >= 0.0); /* sanity check */

    free(b);
    free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("LU CSR Working Format");

    /* Day 1: Conversion tests */
    RUN_TEST(test_lu_csr_null_args);
    RUN_TEST(test_lu_csr_identity);
    RUN_TEST(test_lu_csr_dense_5x5);
    RUN_TEST(test_lu_csr_empty);
    RUN_TEST(test_lu_csr_1x1);
    RUN_TEST(test_lu_csr_with_permutations);
    RUN_TEST(test_lu_csr_fill_factor_clamping);
    RUN_TEST(test_lu_csr_suitesparse_orsirr1);
    RUN_TEST(test_lu_csr_tridiagonal);

    /* Day 2: Elimination tests */
    RUN_TEST(test_lu_csr_eliminate_3x3);
    RUN_TEST(test_lu_csr_eliminate_5x5_solve);
    RUN_TEST(test_lu_csr_eliminate_diagonal);
    RUN_TEST(test_lu_csr_eliminate_tridiag);
    RUN_TEST(test_lu_csr_eliminate_needs_pivot);
    RUN_TEST(test_lu_csr_eliminate_singular);

    /* Day 3: Integration and solve tests */
    RUN_TEST(test_lu_csr_solve_null);
    RUN_TEST(test_lu_csr_factor_solve_tridiag);
    RUN_TEST(test_lu_csr_factor_solve_dense);
    RUN_TEST(test_lu_csr_factor_solve_orsirr1);
    RUN_TEST(test_lu_csr_factor_solve_steam1);
    RUN_TEST(test_lu_csr_benchmark_orsirr1);

    TEST_SUITE_END();
}
