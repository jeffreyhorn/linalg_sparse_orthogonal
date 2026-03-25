#include "sparse_matrix.h"
#include "sparse_cholesky.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * SpMM tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Known 2x2: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]] */
static void test_matmul_2x2(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0); sparse_insert(A, 1, 1, 4.0);

    SparseMatrix *B = sparse_create(2, 2);
    sparse_insert(B, 0, 0, 5.0); sparse_insert(B, 0, 1, 6.0);
    sparse_insert(B, 1, 0, 7.0); sparse_insert(B, 1, 1, 8.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);

    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 19.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 22.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 43.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 50.0, 0.0);

    sparse_free(A); sparse_free(B); sparse_free(C);
}

/* I * A = A */
static void test_matmul_identity_left(void)
{
    idx_t n = 4;
    SparseMatrix *I = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(I, i, i, 1.0);

    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 2.0); sparse_insert(A, 0, 3, 5.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 0, 1.0); sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 3, 7.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(I, A, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), sparse_nnz(A));

    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 3), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 3, 3), 7.0, 0.0);

    sparse_free(I); sparse_free(A); sparse_free(C);
}

/* A * I = A */
static void test_matmul_identity_right(void)
{
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);

    SparseMatrix *I = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(I, i, i, 1.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, I, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), sparse_nnz(A));

    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 0), 7.0, 0.0);

    sparse_free(A); sparse_free(I); sparse_free(C);
}

/* Diagonal * A = scaled rows */
static void test_matmul_diag_left(void)
{
    SparseMatrix *D = sparse_create(2, 2);
    sparse_insert(D, 0, 0, 2.0);
    sparse_insert(D, 1, 1, 3.0);

    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 1, 4.0);
    sparse_insert(A, 1, 0, 5.0); sparse_insert(A, 1, 1, 6.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(D, A, &C), SPARSE_OK);

    /* Row 0 scaled by 2, row 1 scaled by 3 */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 8.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 15.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 18.0, 0.0);

    sparse_free(D); sparse_free(A); sparse_free(C);
}

/* A * Diagonal = scaled columns */
static void test_matmul_diag_right(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 1, 4.0);
    sparse_insert(A, 1, 0, 5.0); sparse_insert(A, 1, 1, 6.0);

    SparseMatrix *D = sparse_create(2, 2);
    sparse_insert(D, 0, 0, 2.0);
    sparse_insert(D, 1, 1, 3.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, D, &C), SPARSE_OK);

    /* Col 0 scaled by 2, col 1 scaled by 3 */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 1), 12.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 10.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 18.0, 0.0);

    sparse_free(A); sparse_free(D); sparse_free(C);
}

/* Dimension mismatch → SPARSE_ERR_SHAPE */
static void test_matmul_shape_mismatch(void)
{
    SparseMatrix *A = sparse_create(2, 3);
    SparseMatrix *B = sparse_create(4, 2);  /* 3 != 4 */
    SparseMatrix *C;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_ERR_SHAPE);
    sparse_free(A); sparse_free(B);
}

/* NULL inputs → SPARSE_ERR_NULL */
static void test_matmul_null(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    SparseMatrix *C;
    ASSERT_ERR(sparse_matmul(NULL, A, &C), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matmul(A, NULL, &C), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_matmul(A, A, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* Rectangular: (3x2) * (2x4) → (3x4) */
static void test_matmul_rectangular(void)
{
    SparseMatrix *A = sparse_create(3, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 0, 3.0); sparse_insert(A, 2, 1, 4.0);

    SparseMatrix *B = sparse_create(2, 4);
    sparse_insert(B, 0, 0, 1.0); sparse_insert(B, 0, 2, 2.0);
    sparse_insert(B, 1, 1, 3.0); sparse_insert(B, 1, 3, 4.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_NOT_NULL(C);
    ASSERT_EQ(sparse_rows(C), 3);
    ASSERT_EQ(sparse_cols(C), 4);

    /* C(0,:) = 1*row0(B) = [1, 0, 2, 0] */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 0, 2), 2.0, 0.0);
    /* C(1,:) = 2*row1(B) = [0, 6, 0, 8] */
    ASSERT_NEAR(sparse_get_phys(C, 1, 1), 6.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 1, 3), 8.0, 0.0);
    /* C(2,:) = 3*row0(B) + 4*row1(B) = [3, 12, 6, 16] */
    ASSERT_NEAR(sparse_get_phys(C, 2, 0), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 1), 12.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 2), 6.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(C, 2, 3), 16.0, 0.0);

    sparse_free(A); sparse_free(B); sparse_free(C);
}

/* Cancellation: A * B where entries cancel to zero → no spurious nnz */
static void test_matmul_cancellation(void)
{
    /* A = [[1, 1]], B = [[1], [-1]] → C = [[0]] */
    SparseMatrix *A = sparse_create(1, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);

    SparseMatrix *B = sparse_create(2, 1);
    sparse_insert(B, 0, 0, 1.0);
    sparse_insert(B, 1, 0, -1.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), 0);  /* cancellation → no entry */

    sparse_free(A); sparse_free(B); sparse_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

/* A * empty = empty (nnz=0) */
static void test_matmul_empty(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);

    SparseMatrix *B = sparse_create(3, 3);  /* all zeros */

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), 0);

    sparse_free(A); sparse_free(B); sparse_free(C);
}

/* Single-row * single-column → 1×1 (dot product) */
static void test_matmul_row_col(void)
{
    SparseMatrix *A = sparse_create(1, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 0, 2, 3.0);

    SparseMatrix *B = sparse_create(3, 1);
    sparse_insert(B, 0, 0, 4.0);
    sparse_insert(B, 1, 0, 5.0);
    sparse_insert(B, 2, 0, 6.0);

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_EQ(sparse_rows(C), 1);
    ASSERT_EQ(sparse_cols(C), 1);
    /* 1*4 + 2*5 + 3*6 = 32 */
    ASSERT_NEAR(sparse_get_phys(C, 0, 0), 32.0, 0.0);

    sparse_free(A); sparse_free(B); sparse_free(C);
}

/* Very sparse: single nnz each */
static void test_matmul_single_nnz(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 1, 2, 5.0);  /* only entry */

    SparseMatrix *B = sparse_create(3, 3);
    sparse_insert(B, 2, 0, 7.0);  /* only entry */

    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(A, B, &C), SPARSE_OK);
    ASSERT_EQ(sparse_nnz(C), 1);
    /* C(1,0) = A(1,2)*B(2,0) = 35 */
    ASSERT_NEAR(sparse_get_phys(C, 1, 0), 35.0, 0.0);

    sparse_free(A); sparse_free(B); sparse_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SuiteSparse validation
 * ═══════════════════════════════════════════════════════════════════════ */

/* Associativity: (A*B)*x = A*(B*x) on nos4 */
static void test_matmul_associativity(void)
{
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/nos4.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    /* Use A as both A and B (A*A) */
    SparseMatrix *AA = NULL;
    ASSERT_ERR(sparse_matmul(A, A, &AA), SPARSE_OK);

    /* x = [1, 2, 3, ...] */
    double *x = malloc((size_t)n * sizeof(double));
    double *Ax = malloc((size_t)n * sizeof(double));
    double *AAx_direct = malloc((size_t)n * sizeof(double));
    double *A_Ax = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(Ax);
    ASSERT_NOT_NULL(AAx_direct);
    ASSERT_NOT_NULL(A_Ax);
    for (idx_t i = 0; i < n; i++) x[i] = (double)(i + 1);

    /* (A*A)*x */
    sparse_matvec(AA, x, AAx_direct);

    /* A*(A*x) */
    sparse_matvec(A, x, Ax);
    sparse_matvec(A, Ax, A_Ax);

    /* Should match */
    double maxdiff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(AAx_direct[i] - A_Ax[i]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("    nos4: (A*A)*x vs A*(A*x) maxdiff = %.2e\n", maxdiff);
    ASSERT_TRUE(maxdiff < 1e-8);

    free(x); free(Ax); free(AAx_direct); free(A_Ax);
    sparse_free(A); sparse_free(AA);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Cholesky integration: L * L^T should reconstruct A
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build L^T from L by manually constructing the transpose */
static SparseMatrix *build_transpose(const SparseMatrix *L, idx_t n)
{
    SparseMatrix *LT = sparse_create(n, n);
    if (!LT) return NULL;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double val = sparse_get_phys(L, i, j);
            if (val != 0.0)
                sparse_insert(LT, j, i, val);
        }
    }
    return LT;
}

static void test_matmul_cholesky_reconstruct(void)
{
    /* A = [[10,1,2],[1,10,1],[2,1,10]] — SPD */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 10.0); sparse_insert(A, 0, 1, 1.0); sparse_insert(A, 0, 2, 2.0);
    sparse_insert(A, 1, 0, 1.0);  sparse_insert(A, 1, 1, 10.0); sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 2.0);  sparse_insert(A, 2, 1, 1.0);  sparse_insert(A, 2, 2, 10.0);

    SparseMatrix *L = sparse_copy(A);
    ASSERT_NOT_NULL(L);
    ASSERT_ERR(sparse_cholesky_factor(L), SPARSE_OK);

    /* Build L^T */
    SparseMatrix *LT = build_transpose(L, 3);
    ASSERT_NOT_NULL(LT);

    /* C = L * L^T should equal A */
    SparseMatrix *C = NULL;
    ASSERT_ERR(sparse_matmul(L, LT, &C), SPARSE_OK);

    /* Verify C ≈ A */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double expected = sparse_get_phys(A, (idx_t)i, (idx_t)j);
            double got = sparse_get_phys(C, (idx_t)i, (idx_t)j);
            ASSERT_NEAR(got, expected, 1e-12);
        }
    }

    sparse_free(A); sparse_free(L); sparse_free(LT); sparse_free(C);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Sparse Matrix-Matrix Multiply Tests");

    RUN_TEST(test_matmul_2x2);
    RUN_TEST(test_matmul_identity_left);
    RUN_TEST(test_matmul_identity_right);
    RUN_TEST(test_matmul_diag_left);
    RUN_TEST(test_matmul_diag_right);
    RUN_TEST(test_matmul_shape_mismatch);
    RUN_TEST(test_matmul_null);
    RUN_TEST(test_matmul_rectangular);
    RUN_TEST(test_matmul_cancellation);

    /* Edge cases */
    RUN_TEST(test_matmul_empty);
    RUN_TEST(test_matmul_row_col);
    RUN_TEST(test_matmul_single_nnz);

    /* SuiteSparse */
    RUN_TEST(test_matmul_associativity);

    /* Cholesky integration */
    RUN_TEST(test_matmul_cholesky_reconstruct);

    TEST_SUITE_END();
}
