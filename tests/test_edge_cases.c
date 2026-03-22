#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_vector.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * 1x1 matrix edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_1x1_complete_lifecycle(void)
{
    /* Create, insert, factor, solve, refine — all on a 1x1 matrix */
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_NOT_NULL(A);
    ASSERT_ERR(sparse_insert(A, 0, 0, 7.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 1);

    double b[] = {21.0};
    double x[1];

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 3.0, 1e-14);

    /* Iterative refinement on 1x1 */
    ASSERT_ERR(sparse_lu_refine(A, LU, b, x, 3, 1e-15), SPARSE_OK);
    ASSERT_NEAR(x[0], 3.0, 1e-15);

    /* Partial pivoting should also work */
    SparseMatrix *LU2 = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU2, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU2, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 3.0, 1e-14);

    sparse_free(LU2);
    sparse_free(LU);
    sparse_free(A);
}

static void test_1x1_extreme_values(void)
{
    /* 1x1 with very large value */
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 1e200);
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    double b[] = {1e200};
    double x[1];
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 1.0, 1e-10);
    sparse_free(LU);
    sparse_free(A);

    /* 1x1 with small-but-above-drop-tol value.
     * Note: backward_sub rejects pivots with |u_ii| < DROP_TOL (1e-14),
     * so values below that threshold are treated as singular. */
    A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 1e-10);
    LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    b[0] = 2e-10;
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 2.0, 1e-6);
    sparse_free(LU);
    sparse_free(A);
}

static void test_1x1_matvec(void)
{
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 42.0);
    double x[] = {3.0};
    double y[1] = {0};
    ASSERT_ERR(sparse_matvec(A, x, y), SPARSE_OK);
    ASSERT_NEAR(y[0], 126.0, 0.0);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Single non-zero element (off-diagonal)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_single_nnz_offdiag_singular(void)
{
    /* A matrix with a single off-diagonal element is singular */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 2, 5.0);
    ASSERT_EQ(sparse_nnz(A), 1);

    SparseMatrix *LU = sparse_copy(A);
    sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12);
    /* After pivot, the 5.0 becomes the (0,0) pivot, but remaining submatrix is zero */
    /* So this should detect singularity at step 1 or 2 */
    ASSERT_ERR(err, SPARSE_ERR_SINGULAR);
    sparse_free(LU);
    sparse_free(A);
}

static void test_single_nnz_diagonal(void)
{
    /* 3x3 with only A(1,1)=5 — rows 0 and 2 have no pivot */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 1, 1, 5.0);

    SparseMatrix *LU = sparse_copy(A);
    sparse_err_t err = sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12);
    ASSERT_ERR(err, SPARSE_ERR_SINGULAR);
    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * All-diagonal matrices with extreme values
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_diagonal_extreme_solve(void)
{
    /*
     * Diagonal with entries spanning moderate orders of magnitude.
     * With relative drop tolerance (DROP_TOL * ||A||_inf), all entries
     * must be large enough relative to the max entry to avoid being
     * flagged as singular. Ratio of 1e8 is well within tolerance.
     */
    idx_t n = 5;
    double diag[] = {1e-4, 1e-2, 1.0, 1e2, 1e4};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);

    /* b = diag, so x should be all 1s */
    double b[5], x[5];
    for (idx_t i = 0; i < n; i++) b[i] = diag[i];

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-10);

    sparse_free(LU);
    sparse_free(A);
}

static void test_diagonal_negative_values(void)
{
    /* Diagonal with negative entries */
    idx_t n = 4;
    double diag[] = {-3.0, 2.0, -1.0, 4.0};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);

    double b[] = {-6.0, 4.0, -2.0, 8.0};  /* x should be [2, 2, 2, 2] */
    double x[4];

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 2.0, 1e-14);

    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Empty matrix (nnz=0) operations
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_empty_matrix_copy(void)
{
    SparseMatrix *A = sparse_create(5, 5);
    ASSERT_EQ(sparse_nnz(A), 0);

    SparseMatrix *B = sparse_copy(A);
    ASSERT_NOT_NULL(B);
    ASSERT_EQ(sparse_nnz(B), 0);
    ASSERT_EQ(sparse_rows(B), 5);
    ASSERT_EQ(sparse_cols(B), 5);

    sparse_free(B);
    sparse_free(A);
}

static void test_empty_matrix_factor_singular(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_err_t err = sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12);
    ASSERT_ERR(err, SPARSE_ERR_SINGULAR);
    sparse_free(A);
}

static void test_empty_matrix_info(void)
{
    SparseMatrix *A = sparse_create(10, 10);
    ASSERT_EQ(sparse_nnz(A), 0);
    ASSERT_TRUE(sparse_memory_usage(A) > 0);  /* struct + headers + perms */

    /* Get on empty returns 0 */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 0.0, 0.0);
    ASSERT_NEAR(sparse_get(A, 5, 5), 0.0, 0.0);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Negative index bounds checking
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_negative_index_remove(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    ASSERT_ERR(sparse_remove(A, -1, 0), SPARSE_ERR_BOUNDS);
    ASSERT_ERR(sparse_remove(A, 0, -1), SPARSE_ERR_BOUNDS);
    ASSERT_ERR(sparse_remove(A, -1, -1), SPARSE_ERR_BOUNDS);
    ASSERT_EQ(sparse_nnz(A), 1);  /* unchanged */
    sparse_free(A);
}

static void test_negative_index_set(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_ERR(sparse_set(A, -1, 0, 5.0), SPARSE_ERR_BOUNDS);
    ASSERT_ERR(sparse_set(A, 0, -1, 5.0), SPARSE_ERR_BOUNDS);
    ASSERT_EQ(sparse_nnz(A), 0);  /* nothing inserted */
    sparse_free(A);
}

static void test_negative_index_get(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 7.0);
    /* get with negative index should return 0.0, not crash */
    ASSERT_NEAR(sparse_get_phys(A, -1, 0), 0.0, 0.0);
    ASSERT_NEAR(sparse_get(A, -1, 0), 0.0, 0.0);
    ASSERT_NEAR(sparse_get(A, 0, -1), 0.0, 0.0);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Large/small values in factor and solve
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_large_value_solve(void)
{
    /* 2x2 system with large coefficients */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1e150);
    sparse_insert(A, 0, 1, 1e149);
    sparse_insert(A, 1, 0, 1e149);
    sparse_insert(A, 1, 1, 1e150);

    /* b = A * [1, 1] */
    double b[2] = {1e150 + 1e149, 1e149 + 1e150};
    double x[2];

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    ASSERT_NEAR(x[0], 1.0, 1e-8);
    ASSERT_NEAR(x[1], 1.0, 1e-8);

    sparse_free(LU);
    sparse_free(A);
}

static void test_small_value_solve(void)
{
    /* 2x2 with small entries (but above DROP_TOL) */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1e-10);
    sparse_insert(A, 1, 1, 2e-10);

    double b[] = {1e-10, 4e-10};  /* x = [1, 2] */
    double x[2];

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    ASSERT_NEAR(x[0], 1.0, 1e-6);
    ASSERT_NEAR(x[1], 2.0, 1e-6);

    sparse_free(LU);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Matrix modification after operations
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_modify_after_copy(void)
{
    /* Ensure copy is truly independent through multiple modifications */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);

    SparseMatrix *B = sparse_copy(A);

    /* Modify original */
    sparse_remove(A, 1, 1);
    sparse_insert(A, 0, 2, 99.0);
    ASSERT_EQ(sparse_nnz(A), 3);

    /* Copy should be unaffected */
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 0, 2), 0.0, 0.0);
    ASSERT_EQ(sparse_nnz(B), 3);

    /* Modify copy */
    sparse_remove(B, 0, 0);
    ASSERT_EQ(sparse_nnz(B), 2);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 1.0, 0.0);  /* original unchanged */

    sparse_free(B);
    sparse_free(A);
}

static void test_insert_remove_insert_same_position(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 1, 1, 5.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 5.0, 0.0);
    ASSERT_EQ(sparse_nnz(A), 1);

    sparse_remove(A, 1, 1);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 0.0, 0.0);
    ASSERT_EQ(sparse_nnz(A), 0);

    sparse_insert(A, 1, 1, 10.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 10.0, 0.0);
    ASSERT_EQ(sparse_nnz(A), 1);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Permutation edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_perms_after_factor_are_valid(void)
{
    /* Verify perm[inv_perm[i]] == i for both row and col after factorization */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    /* 6x6 diag-dominant with some off-diagonal structure */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 10.0);
        if (i > 0)     sparse_insert(A, i, (i + 2) % n, 1.0);
        if (i < n - 1) sparse_insert(A, i, (i + 3) % n, -0.5);
    }

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);

    const idx_t *rp = sparse_row_perm(LU);
    const idx_t *irp = sparse_inv_row_perm(LU);
    const idx_t *cp = sparse_col_perm(LU);
    const idx_t *icp = sparse_inv_col_perm(LU);

    for (idx_t i = 0; i < n; i++) {
        ASSERT_EQ(rp[irp[i]], i);
        ASSERT_EQ(irp[rp[i]], i);
        ASSERT_EQ(cp[icp[i]], i);
        ASSERT_EQ(icp[cp[i]], i);
    }

    sparse_free(LU);
    sparse_free(A);
}

static void test_reset_perms_then_factor(void)
{
    /* Factor, reset perms, re-factor (on a fresh copy) */
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 4.0); sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0); sparse_insert(A, 1, 1, 3.0); sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0); sparse_insert(A, 2, 2, 4.0);

    SparseMatrix *LU1 = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU1, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);

    double b[] = {9.0, 11.0, 13.0};
    double x1[3], x2[3];
    ASSERT_ERR(sparse_lu_solve(LU1, b, x1), SPARSE_OK);

    /* Factor again with partial pivoting */
    SparseMatrix *LU2 = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU2, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(LU2, b, x2), SPARSE_OK);

    /* Both should give the same answer */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x1[i], x2[i], 1e-12);

    sparse_free(LU2);
    sparse_free(LU1);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Stress: many insert/remove cycles (exercises free-list)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_freelist_reuse(void)
{
    /* Insert many elements, remove them all, insert again.
     * The free-list should allow reuse without new slab allocation. */
    SparseMatrix *A = sparse_create(50, 50);
    for (idx_t i = 0; i < 50; i++)
        for (idx_t j = 0; j < 50; j++)
            sparse_insert(A, i, j, (double)(i * 50 + j + 1));
    ASSERT_EQ(sparse_nnz(A), 2500);

    size_t mem_after_fill = sparse_memory_usage(A);

    /* Remove all */
    for (idx_t i = 0; i < 50; i++)
        for (idx_t j = 0; j < 50; j++)
            sparse_remove(A, i, j);
    ASSERT_EQ(sparse_nnz(A), 0);

    /* Re-insert — should reuse freed nodes.
     * Use i+j+1 to avoid inserting 0.0 (which removes instead of inserting). */
    for (idx_t i = 0; i < 50; i++)
        for (idx_t j = 0; j < 50; j++)
            sparse_insert(A, i, j, (double)(i + j + 1));
    ASSERT_EQ(sparse_nnz(A), 2500);

    /* Memory should not have grown (same slabs reused) */
    size_t mem_after_refill = sparse_memory_usage(A);
    ASSERT_EQ(mem_after_fill, mem_after_refill);

    /* Verify values */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 25, 25), 51.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 49, 49), 99.0, 0.0);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Relative tolerance edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_zero_matrix_factor_norm(void)
{
    /* All-zero matrix has norm 0. Factorization should detect singularity
     * during pivot selection (max_val < tol), not in backward sub. */
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12),
               SPARSE_ERR_SINGULAR);
    sparse_free(A);
}

static void test_1x1_tiny_value_solves(void)
{
    /* 1x1 matrix with a tiny value. Relative tolerance should allow it
     * since ||A||_inf == |val|, so threshold = DROP_TOL * |val| << |val|. */
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 1e-300);
    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_PARTIAL, 1e-308), SPARSE_OK);

    double b = 1e-300, x;
    ASSERT_ERR(sparse_lu_solve(A, &b, &x), SPARSE_OK);
    ASSERT_NEAR(x, 1.0, 1e-10);
    sparse_free(A);
}

static void test_mixed_scale_no_false_singular(void)
{
    /* 3x3 with mixed but not extreme scales. All entries are well above
     * relative threshold. Should solve cleanly. */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 0.5);
    sparse_insert(A, 1, 0, 0.5);
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 1, 2, 0.5);
    sparse_insert(A, 2, 1, 0.5);
    sparse_insert(A, 2, 2, 1.0);

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    double b[] = {1.0, 1.0, 1.0};
    double x[3];
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Verify via residual */
    double r[3];
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < 3; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    sparse_free(A);
    sparse_free(LU);
}

static void test_large_norm_does_not_mask_singularity(void)
{
    /* Matrix with large entries but a zero row → singular.
     * Large ||A||_inf makes relative threshold large, but pivot selection
     * should catch the zero pivot before backward sub is reached. */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1e15);
    sparse_insert(A, 0, 1, 1e15);
    /* row 1 is all zeros → singular */
    sparse_insert(A, 2, 2, 1e15);

    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12),
               SPARSE_ERR_SINGULAR);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Edge Case & Hardening Tests");

    /* 1x1 matrices */
    RUN_TEST(test_1x1_complete_lifecycle);
    RUN_TEST(test_1x1_extreme_values);
    RUN_TEST(test_1x1_matvec);

    /* Single non-zero */
    RUN_TEST(test_single_nnz_offdiag_singular);
    RUN_TEST(test_single_nnz_diagonal);

    /* Diagonal extremes */
    RUN_TEST(test_diagonal_extreme_solve);
    RUN_TEST(test_diagonal_negative_values);

    /* Empty matrix */
    RUN_TEST(test_empty_matrix_copy);
    RUN_TEST(test_empty_matrix_factor_singular);
    RUN_TEST(test_empty_matrix_info);

    /* Negative indices */
    RUN_TEST(test_negative_index_remove);
    RUN_TEST(test_negative_index_set);
    RUN_TEST(test_negative_index_get);

    /* Large/small values */
    RUN_TEST(test_large_value_solve);
    RUN_TEST(test_small_value_solve);

    /* Modification */
    RUN_TEST(test_modify_after_copy);
    RUN_TEST(test_insert_remove_insert_same_position);

    /* Permutations */
    RUN_TEST(test_perms_after_factor_are_valid);
    RUN_TEST(test_reset_perms_then_factor);

    /* Free-list stress */
    RUN_TEST(test_freelist_reuse);

    /* Relative tolerance edge cases */
    RUN_TEST(test_zero_matrix_factor_norm);
    RUN_TEST(test_1x1_tiny_value_solves);
    RUN_TEST(test_mixed_scale_no_false_singular);
    RUN_TEST(test_large_norm_does_not_mask_singularity);

    TEST_SUITE_END();
}
