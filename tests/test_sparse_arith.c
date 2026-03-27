#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_scale tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_scale_identity_by_3(void) {
    SparseMatrix *A = sparse_create(4, 4);
    for (idx_t i = 0; i < 4; i++)
        sparse_insert(A, i, i, 1.0);

    ASSERT_ERR(sparse_scale(A, 3.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 4);
    for (idx_t i = 0; i < 4; i++)
        ASSERT_NEAR(sparse_get_phys(A, i, i), 3.0, 0.0);

    sparse_free(A);
}

static void test_scale_by_1(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 5.0);
    sparse_insert(A, 1, 2, 7.0);
    sparse_insert(A, 2, 1, -3.0);

    ASSERT_ERR(sparse_scale(A, 1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 3);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 2), 7.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 1), -3.0, 0.0);

    sparse_free(A);
}

static void test_scale_by_0(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);
    ASSERT_EQ(sparse_nnz(A), 3);

    ASSERT_ERR(sparse_scale(A, 0.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 0);

    /* All entries should be gone */
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++)
            ASSERT_NEAR(sparse_get_phys(A, i, j), 0.0, 0.0);

    sparse_free(A);
}

static void test_scale_by_negative_1(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, -2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 1.0);

    ASSERT_ERR(sparse_scale(A, -1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 4);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), -4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 0), -3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), -1.0, 0.0);

    sparse_free(A);
}

static void test_scale_rectangular(void) {
    SparseMatrix *A = sparse_create(2, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 4, 2.0);
    sparse_insert(A, 1, 2, 3.0);

    ASSERT_ERR(sparse_scale(A, 0.5), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 3);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 0.5, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 4), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 2), 1.5, 0.0);

    sparse_free(A);
}

static void test_scale_null(void) { ASSERT_ERR(sparse_scale(NULL, 2.0), SPARSE_ERR_NULL); }

static void test_scale_empty_matrix(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_ERR(sparse_scale(A, 5.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 0);
    sparse_free(A);
}

static void test_scale_invalidates_norm(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);

    double norm;
    ASSERT_ERR(sparse_norminf(A, &norm), SPARSE_OK);
    ASSERT_NEAR(norm, 2.0, 0.0);

    ASSERT_ERR(sparse_scale(A, 3.0), SPARSE_OK);

    ASSERT_ERR(sparse_norminf(A, &norm), SPARSE_OK);
    ASSERT_NEAR(norm, 6.0, 0.0);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_add tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_add_a_plus_zero(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);

    SparseMatrix *Z = sparse_create(3, 3); /* zero matrix */

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, Z, 1.0, 1.0, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_nnz(C), 3);
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 3.0, 0.0);

    sparse_free(A);
    sparse_free(Z);
    sparse_free(C);
}

static void test_add_a_minus_a(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 5.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 7.0);
    sparse_insert(A, 2, 0, -1.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, A, 1.0, -1.0, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_nnz(C), 0); /* everything cancels */

    sparse_free(A);
    sparse_free(C);
}

static void test_add_disjoint_patterns(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 2, 2, 3.0);

    SparseMatrix *B = sparse_create(3, 3);
    sparse_insert(B, 0, 2, 4.0);
    sparse_insert(B, 1, 1, 5.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), 4);
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 3.0, 0.0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_add_overlapping_entries(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);

    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 7.0);
    sparse_insert(B, 1, 1, 6.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, B, 2.0, 0.5, &C), SPARSE_OK);
    /* C = 2*A + 0.5*B = (6+3.5, 0; 0, 8+3) = (9.5, 0; 0, 11) */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 9.5, 1e-14);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 11.0, 1e-14);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_add_dimension_mismatch(void) {
    SparseMatrix *A = sparse_create(3, 3);
    SparseMatrix *B = sparse_create(3, 4);
    SparseMatrix *C = NULL;

    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_ERR_SHAPE);
    ASSERT_NULL(C);

    sparse_free(A);
    sparse_free(B);
}

static void test_add_rectangular(void) {
    SparseMatrix *A = sparse_create(2, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 3, 2.0);

    SparseMatrix *B = sparse_create(2, 4);
    sparse_insert(B, 0, 0, 3.0);
    sparse_insert(B, 0, 2, 4.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);
    ASSERT_EQ(sparse_rows(C), 2);
    ASSERT_EQ(sparse_cols(C), 4);
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 3), 2.0, 0.0);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

static void test_add_null_args(void) {
    SparseMatrix *A = sparse_create(2, 2);
    SparseMatrix *C = NULL;

    ASSERT_ERR(sparse_add(NULL, A, 1.0, 1.0, &C), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_add(A, NULL, 1.0, 1.0, &C), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_add(A, A, 1.0, 1.0, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_add_inplace tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_add_inplace_basic(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);

    SparseMatrix *B = sparse_create(3, 3);
    sparse_insert(B, 0, 0, 3.0);
    sparse_insert(B, 2, 2, 4.0);

    ASSERT_ERR(sparse_add_inplace(A, B, 1.0, 1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 3);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 2), 4.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

static void test_add_inplace_with_scaling(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 10.0);
    sparse_insert(A, 1, 1, 20.0);

    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 1.0);
    sparse_insert(B, 1, 1, 2.0);

    /* A = 0.5*A + 3*B = (5+3, 0; 0, 10+6) = (8, 0; 0, 16) */
    ASSERT_ERR(sparse_add_inplace(A, B, 0.5, 3.0), SPARSE_OK);
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 8.0, 1e-14);
    ASSERT_NEAR(sparse_get_phys(A, 1, 1), 16.0, 1e-14);

    sparse_free(A);
    sparse_free(B);
}

static void test_add_inplace_cancellation(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 5.0);
    sparse_insert(A, 1, 1, 3.0);

    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 5.0);
    sparse_insert(B, 1, 1, 3.0);

    ASSERT_ERR(sparse_add_inplace(A, B, 1.0, -1.0), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(A), 0);

    sparse_free(A);
    sparse_free(B);
}

static void test_add_inplace_null(void) {
    SparseMatrix *A = sparse_create(2, 2);
    ASSERT_ERR(sparse_add_inplace(NULL, A, 1.0, 1.0), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_add_inplace(A, NULL, 1.0, 1.0), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_add_inplace_shape_mismatch(void) {
    SparseMatrix *A = sparse_create(2, 2);
    SparseMatrix *B = sparse_create(3, 3);
    ASSERT_ERR(sparse_add_inplace(A, B, 1.0, 1.0), SPARSE_ERR_SHAPE);
    sparse_free(A);
    sparse_free(B);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Integration: arithmetic + solver
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_add_then_factor_solve(void) {
    /*
     * Construct A = 4*I + T where T is tridiagonal (-1, 0, -1).
     * This gives a diagonally dominant 5x5 tridiagonal.
     */
    idx_t n = 5;

    /* I: identity */
    SparseMatrix *I = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(I, i, i, 1.0);

    /* T: off-diagonal tridiagonal */
    SparseMatrix *T = sparse_create(n, n);
    for (idx_t i = 0; i < n - 1; i++) {
        sparse_insert(T, i, i + 1, -1.0);
        sparse_insert(T, i + 1, i, -1.0);
    }

    /* A = 4*I + 1*T */
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_add(I, T, 4.0, 1.0, &A), SPARSE_OK);

    /* Verify: diagonal = 4, off-diag = -1 */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 0, 1), -1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(A, 2, 2), 4.0, 0.0);

    /* Copy for residual, factor, solve */
    SparseMatrix *A_orig = sparse_copy(A);
    ASSERT_NOT_NULL(A_orig);
    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    double b[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double x[5];
    ASSERT_ERR(sparse_lu_solve(A, b, x), SPARSE_OK);

    /* Check residual ||b - A_orig*x|| */
    double r[5];
    sparse_matvec(A_orig, x, r);
    double max_res = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = fabs(b[i] - r[i]);
        if (ri > max_res)
            max_res = ri;
    }
    ASSERT_TRUE(max_res < 1e-12);

    sparse_free(I);
    sparse_free(T);
    sparse_free(A);
    sparse_free(A_orig);
}

static void test_scale_then_factor_solve(void) {
    /* Build 3x3 matrix, scale by 0.001, factor and solve */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4000.0);
    sparse_insert(A, 0, 1, 1000.0);
    sparse_insert(A, 1, 0, 1000.0);
    sparse_insert(A, 1, 1, 3000.0);
    sparse_insert(A, 1, 2, 1000.0);
    sparse_insert(A, 2, 1, 1000.0);
    sparse_insert(A, 2, 2, 2000.0);

    ASSERT_ERR(sparse_scale(A, 0.001), SPARSE_OK);

    /* Now entries are 4, 1, 1, 3, 1, 1, 2 */
    ASSERT_NEAR(sparse_get_phys(A, 0, 0), 4.0, 1e-14);

    SparseMatrix *A_orig = sparse_copy(A);
    ASSERT_NOT_NULL(A_orig);
    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);

    double b[] = {1.0, 1.0, 1.0};
    double x[3];
    ASSERT_ERR(sparse_lu_solve(A, b, x), SPARSE_OK);

    /* Residual check */
    double r[3];
    sparse_matvec(A_orig, x, r);
    for (idx_t i = 0; i < 3; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    sparse_free(A);
    sparse_free(A_orig);
}

static void test_norminf_after_add(void) {
    /* Verify norminf works correctly on a matrix produced by sparse_add */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 1, 3.0);

    SparseMatrix *B = sparse_create(3, 3);
    sparse_insert(B, 0, 0, 4.0);
    sparse_insert(B, 2, 2, 5.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_add(A, B, 1.0, 1.0, &C), SPARSE_OK);

    /* C: row 0 = [5, 2, 0] sum=7, row 1 = [0, 3, 0] sum=3, row 2 = [0, 0, 5] sum=5 */
    double norm;
    ASSERT_ERR(sparse_norminf(C, &norm), SPARSE_OK);
    ASSERT_NEAR(norm, 7.0, 1e-14);

    sparse_free(A);
    sparse_free(B);
    sparse_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Sparse Matrix Arithmetic Tests");

    /* Scale */
    RUN_TEST(test_scale_identity_by_3);
    RUN_TEST(test_scale_by_1);
    RUN_TEST(test_scale_by_0);
    RUN_TEST(test_scale_by_negative_1);
    RUN_TEST(test_scale_rectangular);
    RUN_TEST(test_scale_null);
    RUN_TEST(test_scale_empty_matrix);
    RUN_TEST(test_scale_invalidates_norm);

    /* Add (out-of-place) */
    RUN_TEST(test_add_a_plus_zero);
    RUN_TEST(test_add_a_minus_a);
    RUN_TEST(test_add_disjoint_patterns);
    RUN_TEST(test_add_overlapping_entries);
    RUN_TEST(test_add_dimension_mismatch);
    RUN_TEST(test_add_rectangular);
    RUN_TEST(test_add_null_args);

    /* Add (in-place) */
    RUN_TEST(test_add_inplace_basic);
    RUN_TEST(test_add_inplace_with_scaling);
    RUN_TEST(test_add_inplace_cancellation);
    RUN_TEST(test_add_inplace_null);
    RUN_TEST(test_add_inplace_shape_mismatch);

    /* Integration: arithmetic + solver */
    RUN_TEST(test_add_then_factor_solve);
    RUN_TEST(test_scale_then_factor_solve);
    RUN_TEST(test_norminf_after_add);

    TEST_SUITE_END();
}
