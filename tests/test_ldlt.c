#include "sparse_cholesky.h"
#include "sparse_ldlt.h"
#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DATA_DIR "tests/data"
#define SS_DIR DATA_DIR "/suitesparse"

/* ═══════════════════════════════════════════════════════════════════════
 * Entry validation tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ldlt_null_args(void) {
    sparse_ldlt_t ldlt;
    SparseMatrix *A = sparse_create(3, 3);

    ASSERT_ERR(sparse_ldlt_factor(NULL, &ldlt), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ldlt_factor(A, NULL), SPARSE_ERR_NULL);

    ASSERT_ERR(sparse_ldlt_solve(NULL, NULL, NULL), SPARSE_ERR_NULL);

    sparse_ldlt_free(NULL); /* should not crash */

    sparse_free(A);
}

static void test_ldlt_non_square(void) {
    SparseMatrix *A = sparse_create(3, 4);
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_ERR_SHAPE);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_non_symmetric(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0); /* asymmetric: A(0,1) != A(1,0) */
    sparse_insert(A, 1, 1, 1.0);
    sparse_insert(A, 2, 2, 1.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_ERR_NOT_SPD);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_empty_matrix(void) {
    /* sparse_create(0,0) returns NULL on this library, so test that
     * a valid 1x1 zero matrix is detected as singular instead. */
    SparseMatrix *A = sparse_create(1, 1);
    /* A(0,0) = 0 → singular */
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_ERR_SINGULAR);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 7.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);
    ASSERT_EQ(ldlt.n, 1);
    ASSERT_NEAR(ldlt.D[0], 7.0, 1e-14);
    ASSERT_EQ(ldlt.pivot_size[0], 1);

    /* Solve: 7*x = 14 → x = 2 */
    double b = 14.0, x;
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, &b, &x), SPARSE_OK);
    ASSERT_NEAR(x, 2.0, 1e-14);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_1x1_negative(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, -5.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);
    ASSERT_NEAR(ldlt.D[0], -5.0, 1e-14);

    /* Inertia: (0, 1, 0) */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 0);
    ASSERT_EQ(neg, 1);
    ASSERT_EQ(zero, 0);

    /* Solve: -5*x = 10 → x = -2 */
    double b = 10.0, x;
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, &b, &x), SPARSE_OK);
    ASSERT_NEAR(x, -2.0, 1e-14);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Diagonal matrix tests (no fill-in, 1x1 pivots only)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ldlt_diagonal_spd(void) {
    /* Diagonal SPD: A = diag(4, 3, 2)
     * LDL^T: L = I, D = diag(4, 3, 2) */
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 2.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Check D */
    ASSERT_NEAR(ldlt.D[0], 4.0, 1e-14);
    ASSERT_NEAR(ldlt.D[1], 3.0, 1e-14);
    ASSERT_NEAR(ldlt.D[2], 2.0, 1e-14);

    /* Check L is identity (nnz = n diagonal entries only) */
    ASSERT_EQ(sparse_nnz(ldlt.L), n);

    /* Inertia: (3, 0, 0) */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 3);
    ASSERT_EQ(neg, 0);
    ASSERT_EQ(zero, 0);

    /* Solve: diag(4,3,2) * x = [8,9,6] → x = [2,3,3] */
    double b[] = {8.0, 9.0, 6.0};
    double x[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 2.0, 1e-14);
    ASSERT_NEAR(x[1], 3.0, 1e-14);
    ASSERT_NEAR(x[2], 3.0, 1e-14);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_diagonal_indefinite(void) {
    /* Diagonal indefinite: A = diag(3, -2, 1, -4) */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, -2.0);
    sparse_insert(A, 2, 2, 1.0);
    sparse_insert(A, 3, 3, -4.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Inertia: (2, 2, 0) */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 2);
    ASSERT_EQ(neg, 2);
    ASSERT_EQ(zero, 0);

    /* Solve: diag(3,-2,1,-4) * x = [6,-4,5,-8] → x = [2,2,5,2] */
    double b[] = {6.0, -4.0, 5.0, -8.0};
    double x[4];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 2.0, 1e-14);
    ASSERT_NEAR(x[1], 2.0, 1e-14);
    ASSERT_NEAR(x[2], 5.0, 1e-14);
    ASSERT_NEAR(x[3], 2.0, 1e-14);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_singular_diagonal(void) {
    /* Singular: A = diag(1, 0, 1) → should fail */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    /* A(1,1) = 0 */
    sparse_insert(A, 2, 2, 1.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_ERR_SINGULAR);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 1x1 pivot factorization tests (SPD — Day 3)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ldlt_spd_3x3(void) {
    /* A = [4 1 0; 1 3 1; 0 1 4] — SPD tridiagonal
     * LDL^T should produce correct L and D such that L*D*L^T = A */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Verify via solve: A*x = b where b = A*[1,1,1] = [5,5,5] */
    double b[] = {5.0, 5.0, 5.0};
    double x[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-12);

    /* Verify residual */
    double r[3];
    sparse_matvec(A, x, r);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    /* Inertia should be (3, 0, 0) for SPD */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 3);
    ASSERT_EQ(neg, 0);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_spd_5x5_tridiag(void) {
    /* 5x5 SPD tridiagonal: diag=4, off=-1 */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Solve A*x = b where b = A*ones */
    double ones[] = {1, 1, 1, 1, 1};
    double b[5], x[5];
    sparse_matvec(A, ones, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_identity(void) {
    /* A = I₃ → L = I, D = I */
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(ldlt.D[i], 1.0, 1e-14);
        ASSERT_EQ(ldlt.pivot_size[i], 1);
    }
    /* L should be identity (only diagonal entries) */
    ASSERT_EQ(sparse_nnz(ldlt.L), n);

    double b[] = {3.0, 7.0, 2.0}, x[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 3.0, 1e-14);
    ASSERT_NEAR(x[1], 7.0, 1e-14);
    ASSERT_NEAR(x[2], 2.0, 1e-14);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_reconstruct_ldlt(void) {
    /* Verify L*D*L^T = A by computing the product explicitly.
     * A = [10 2 3; 2 8 1; 3 1 6] (SPD, dense) */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 10.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 8.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 0, 3.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 6.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Reconstruct: for each (i,j), compute sum_k L(i,k)*D[k]*L(j,k) */
    idx_t n = 3;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double sum = 0.0;
            for (idx_t k = 0; k < n; k++) {
                double l_ik = (i == k) ? 1.0 : sparse_get_phys(ldlt.L, i, k);
                double l_jk = (j == k) ? 1.0 : sparse_get_phys(ldlt.L, j, k);
                sum += l_ik * ldlt.D[k] * l_jk;
            }
            double a_ij = sparse_get_phys(A, i, j);
            ASSERT_NEAR(sum, a_ij, 1e-12);
        }
    }

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_solve_multiple_rhs(void) {
    /* Factor once, solve twice with different RHS */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* RHS 1: b = [5,5,5] → x = [1,1,1] */
    double b1[] = {5.0, 5.0, 5.0}, x1[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b1, x1), SPARSE_OK);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x1[i], 1.0, 1e-12);

    /* RHS 2: b = [4,3,4] */
    double b2[] = {4.0, 3.0, 4.0}, x2[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b2, x2), SPARSE_OK);
    /* Verify via residual */
    double r[3];
    sparse_matvec(A, x2, r);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(r[i], b2[i], 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_free_zeroed(void) {
    /* sparse_ldlt_free on zeroed struct should not crash */
    sparse_ldlt_t ldlt;
    memset(&ldlt, 0, sizeof(ldlt));
    sparse_ldlt_free(&ldlt);
}

static void test_ldlt_inertia_null(void) {
    ASSERT_ERR(sparse_ldlt_inertia(NULL, NULL, NULL, NULL), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 2x2 pivot tests (Bunch-Kaufman — Day 4)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ldlt_2x2_indefinite(void) {
    /* A = [[0,1],[1,0]] — must use 2x2 pivot (zero diagonal) */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Should be a single 2x2 pivot */
    ASSERT_EQ(ldlt.pivot_size[0], 2);
    ASSERT_EQ(ldlt.pivot_size[1], 2);
    ASSERT_NEAR(ldlt.D_offdiag[0], 1.0, 1e-14);

    /* Inertia: one positive, one negative eigenvalue */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 1);
    ASSERT_EQ(neg, 1);
    ASSERT_EQ(zero, 0);

    /* Solve: A*x = [1,0] → x = [0,1] */
    double b[] = {1.0, 0.0};
    double x[2];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 0.0, 1e-14);
    ASSERT_NEAR(x[1], 1.0, 1e-14);

    /* Verify residual */
    double r[2];
    sparse_matvec(A, x, r);
    ASSERT_NEAR(r[0], b[0], 1e-14);
    ASSERT_NEAR(r[1], b[1], 1e-14);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_2x2_with_trailing(void) {
    /* A = [[0,3,1],[3,0,2],[1,2,5]] — 2x2 pivot at (0,1), 1x1 at (2,2)
     * Has nontrivial L entries below the 2x2 block. */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 0.0);
    sparse_insert(A, 0, 1, 3.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 0.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 1, 2.0);
    sparse_insert(A, 2, 2, 5.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* First block should be 2x2 */
    ASSERT_EQ(ldlt.pivot_size[0], 2);
    ASSERT_EQ(ldlt.pivot_size[1], 2);
    ASSERT_EQ(ldlt.pivot_size[2], 1);

    /* Verify D block */
    ASSERT_NEAR(ldlt.D[0], 0.0, 1e-14);
    ASSERT_NEAR(ldlt.D[1], 0.0, 1e-14);
    ASSERT_NEAR(ldlt.D_offdiag[0], 3.0, 1e-14);

    /* Inertia: det of 2x2 block = -9 < 0 → 1 pos + 1 neg; D[2] > 0 → 1 pos
     * Total: 2 positive, 1 negative */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 2);
    ASSERT_EQ(neg, 1);
    ASSERT_EQ(zero, 0);

    /* Solve A*x = [1,1,1] and verify residual */
    double b[] = {1.0, 1.0, 1.0};
    double x[3], r[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_2x2_reconstruct(void) {
    /* Verify L*D*L^T = P*A*P^T for the 2x2 pivot case.
     * A = [[0,3,1],[3,0,2],[1,2,5]] */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 0.0);
    sparse_insert(A, 0, 1, 3.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 0.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 1, 2.0);
    sparse_insert(A, 2, 2, 5.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Build P*A*P^T in dense form */
    idx_t n = 3;
    double PA[3][3];
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++)
            PA[i][j] = sparse_get_phys(A, ldlt.perm[i], ldlt.perm[j]);

    /* Reconstruct L*D*L^T handling 2x2 blocks */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j <= i; j++) {
            double sum = 0.0;
            idx_t blk = 0;
            while (blk < n) {
                double l_i_b = (i == blk) ? 1.0 : sparse_get_phys(ldlt.L, i, blk);
                double l_j_b = (j == blk) ? 1.0 : sparse_get_phys(ldlt.L, j, blk);
                if (ldlt.pivot_size[blk] == 1) {
                    sum += l_i_b * ldlt.D[blk] * l_j_b;
                    blk++;
                } else {
                    double l_i_b1 = (i == blk + 1) ? 1.0 : sparse_get_phys(ldlt.L, i, blk + 1);
                    double l_j_b1 = (j == blk + 1) ? 1.0 : sparse_get_phys(ldlt.L, j, blk + 1);
                    double d11 = ldlt.D[blk];
                    double d22 = ldlt.D[blk + 1];
                    double d21 = ldlt.D_offdiag[blk];
                    /* [l_i_b l_i_b1] * [[d11 d21],[d21 d22]] * [l_j_b; l_j_b1] */
                    sum += l_i_b * (d11 * l_j_b + d21 * l_j_b1) +
                           l_i_b1 * (d21 * l_j_b + d22 * l_j_b1);
                    blk += 2;
                }
            }
            ASSERT_NEAR(sum, PA[i][j], 1e-12);
        }
    }

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_mixed_pivots_4x4(void) {
    /* A = [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,2]]
     * Should use 1x1 at (0,0), 2x2 at (1,2), 1x1 at (3,3) */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 3, 3, 2.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    ASSERT_EQ(ldlt.pivot_size[0], 1);
    ASSERT_EQ(ldlt.pivot_size[1], 2);
    ASSERT_EQ(ldlt.pivot_size[2], 2);
    ASSERT_EQ(ldlt.pivot_size[3], 1);

    /* Inertia: 1 from (0,0)>0, 1+1 from 2x2 det<0, 1 from (3,3)>0 → (2,1,1)?
     * No — det of 2x2 block = 0*0-1 = -1 < 0 → 1 pos + 1 neg.
     * D[0]=1>0, D[3]=2>0 → total (3, 1, 0) */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 3);
    ASSERT_EQ(neg, 1);
    ASSERT_EQ(zero, 0);

    /* Solve and verify residual */
    double b[] = {2.0, 3.0, 4.0, 6.0};
    double x[4], r[4];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_2x2_bk_swap(void) {
    /* A = [[0,1,2],[1,0,3],[2,3,5]]
     * Zero diagonal at (0,0) and (1,1) — BK should swap or use 2x2 pivot.
     * Verify solve correctness regardless of pivot strategy chosen. */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 0.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 2.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 0.0);
    sparse_insert(A, 1, 2, 3.0);
    sparse_insert(A, 2, 0, 2.0);
    sparse_insert(A, 2, 1, 3.0);
    sparse_insert(A, 2, 2, 5.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Solve A*x = [3,4,10] and verify residual */
    double b[] = {3.0, 4.0, 10.0};
    double x[3], r[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    /* Also verify with b = A*[1,1,1] = [3,4,10] */
    ASSERT_NEAR(x[0], 1.0, 1e-12);
    ASSERT_NEAR(x[1], 1.0, 1e-12);
    ASSERT_NEAR(x[2], 1.0, 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_2x2_kkt_like(void) {
    /* Small KKT-like saddle-point matrix:
     * H = [[4 1],[1 3]], A_eq = [[1 0],[0 1]], so
     * K = [[4 1 1 0],
     *      [1 3 0 1],
     *      [1 0 0 0],
     *      [0 1 0 0]]
     * This is symmetric indefinite with inertia (2,2,0). */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 3, 1.0);
    sparse_insert(A, 2, 0, 1.0);
    sparse_insert(A, 2, 2, 0.0);
    sparse_insert(A, 3, 1, 1.0);
    sparse_insert(A, 3, 3, 0.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Inertia should be (2, 2, 0) for a KKT matrix with n=2, m=2 */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos + neg + zero, n);
    ASSERT_EQ(pos, 2);
    ASSERT_EQ(neg, 2);

    /* Solve K*x = [1,1,1,1] and verify residual */
    double b[] = {1.0, 1.0, 1.0, 1.0};
    double x[4], r[4];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Hardening & edge case tests (Day 5)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ldlt_singular_zero_block(void) {
    /* A = [[0,0],[0,0]] — completely zero, must detect singular.
     * The 2x2 block has det = 0, and the matrix has no nonzeros. */
    SparseMatrix *A = sparse_create(2, 2);
    /* All entries are zero (nothing inserted) */

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_ERR_SINGULAR);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_singular_rank_deficient(void) {
    /* A = [[1,1],[1,1]] — rank 1, should fail as singular.
     * D[0] = 1, L(1,0) = 1, then D[1] = 1 - 1*1*1 = 0 → singular. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_ERR_SINGULAR);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_multiple_2x2_blocks(void) {
    /* 6x6 block-diagonal with three 2x2 indefinite blocks:
     * A = blkdiag([[0,2],[2,0]], [[0,3],[3,0]], [[0,1],[1,0]])
     * Each block requires a 2x2 pivot.  All blocks are nonsingular. */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    /* Block 1: (0,1) */
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0);
    /* Block 2: (2,3) */
    sparse_insert(A, 2, 3, 3.0);
    sparse_insert(A, 3, 2, 3.0);
    /* Block 3: (4,5) */
    sparse_insert(A, 4, 5, 1.0);
    sparse_insert(A, 5, 4, 1.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* All should be 2x2 pivots */
    for (idx_t i = 0; i < n; i++)
        ASSERT_EQ(ldlt.pivot_size[i], 2);

    /* Inertia: each 2x2 block contributes (1+,1-) → (3,3,0) */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 3);
    ASSERT_EQ(neg, 3);
    ASSERT_EQ(zero, 0);

    /* Solve and verify residual */
    double b[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double x[6], r[6];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 6; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_large_indefinite(void) {
    /* 10x10 arrow-shaped symmetric indefinite matrix:
     * A(i,i) = (i % 2 == 0) ? 1.0 : -1.0  (alternating signs)
     * A(i,0) = A(0,i) = 0.5 for i > 0  (arrow pattern)
     * This is nonsingular and indefinite with many pivots. */
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        double diag = (i % 2 == 0) ? 4.0 : -4.0;
        sparse_insert(A, i, i, diag);
        if (i > 0) {
            sparse_insert(A, i, 0, 0.5);
            sparse_insert(A, 0, i, 0.5);
        }
    }

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Solve A*x = ones and verify residual */
    double b[10], x[10], r[10];
    for (int i = 0; i < 10; i++)
        b[i] = 1.0;
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 10; i++)
        ASSERT_NEAR(r[i], b[i], 1e-10);

    /* Check inertia sums to n */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos + neg + zero, n);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_tridiag_indefinite(void) {
    /* 8x8 tridiagonal indefinite: diag = [1,-2,3,-4,5,-6,7,-8],
     * off-diag = 0.5.  Tests many elimination steps with alternating signs. */
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        double diag = (double)(i + 1) * ((i % 2 == 0) ? 1.0 : -1.0);
        sparse_insert(A, i, i, diag);
        if (i > 0) {
            sparse_insert(A, i, i - 1, 0.5);
            sparse_insert(A, i - 1, i, 0.5);
        }
    }

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Solve and verify residual */
    double b[8], x[8], r[8];
    for (int i = 0; i < 8; i++)
        b[i] = (double)(i + 1);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 8; i++)
        ASSERT_NEAR(r[i], b[i], 1e-10);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_error_recovery(void) {
    /* After a failed factorization, sparse_ldlt_free must not crash
     * and the struct should be in a clean state. */
    SparseMatrix *A = sparse_create(3, 3);
    /* Singular: rank-1 matrix [[1,2,3],[2,4,6],[3,6,9]] */
    double vals[] = {1, 2, 3};
    for (idx_t i = 0; i < 3; i++)
        for (idx_t j = 0; j < 3; j++)
            sparse_insert(A, i, j, vals[i] * vals[j]);

    sparse_ldlt_t ldlt;
    memset(&ldlt, 0, sizeof(ldlt));
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_ERR_SINGULAR);

    /* After failure, struct should be zeroed (safe to free again) */
    ASSERT_NULL(ldlt.L);
    ASSERT_NULL(ldlt.D);
    ASSERT_NULL(ldlt.perm);

    /* Double-free safety */
    sparse_ldlt_free(&ldlt);

    sparse_free(A);
}

static void test_ldlt_2x2_with_reorder(void) {
    /* Test that 2x2 pivots work correctly with AMD reordering.
     * A = [[0,1,0,2],[1,0,0,3],[0,0,5,1],[2,3,1,0]]
     * This is indefinite with off-diagonal coupling. */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 0, 3, 2.0);
    sparse_insert(A, 3, 0, 2.0);
    sparse_insert(A, 1, 3, 3.0);
    sparse_insert(A, 3, 1, 3.0);
    sparse_insert(A, 2, 2, 5.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 2, 1.0);

    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor_opts(A, &opts, &ldlt), SPARSE_OK);

    /* Solve and verify residual */
    double b[] = {1.0, 2.0, 3.0, 4.0};
    double x[4], r[4];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(r[i], b[i], 1e-11);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_scaled_matrix(void) {
    /* Test tolerance scaling: same structure as 2x2 test but scaled
     * by 1e8 — should still factorize correctly. */
    double s = 1e8;
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 1, 3.0 * s);
    sparse_insert(A, 0, 2, 1.0 * s);
    sparse_insert(A, 1, 0, 3.0 * s);
    sparse_insert(A, 1, 2, 2.0 * s);
    sparse_insert(A, 2, 0, 1.0 * s);
    sparse_insert(A, 2, 1, 2.0 * s);
    sparse_insert(A, 2, 2, 5.0 * s);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Solve A*x = b where b = A*[1,1,1] */
    double b[3], x[3], r[3];
    double ones[] = {1.0, 1.0, 1.0};
    sparse_matvec(A, ones, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(r[i], b[i], 1e-4 * s);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Solve tests (Day 6)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ldlt_solve_vs_cholesky(void) {
    /* Factor the same SPD matrix with both LDL^T and Cholesky.
     * Solutions must agree to near machine precision. */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    /* SPD 4x4: A = [[10,1,0,2],[1,8,1,0],[0,1,6,1],[2,0,1,9]] */
    sparse_insert(A, 0, 0, 10.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 3, 2.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 8.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 6.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 0, 2.0);
    sparse_insert(A, 3, 2, 1.0);
    sparse_insert(A, 3, 3, 9.0);

    /* LDL^T solve */
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);
    double b[] = {7.0, 3.0, 5.0, 2.0};
    double x_ldlt[4];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x_ldlt), SPARSE_OK);

    /* Cholesky solve (needs a copy since it factors in-place) */
    SparseMatrix *Acopy = sparse_copy(A);
    ASSERT_ERR(sparse_cholesky_factor(Acopy), SPARSE_OK);
    double x_chol[4];
    ASSERT_ERR(sparse_cholesky_solve(Acopy, b, x_chol), SPARSE_OK);

    /* Solutions must agree */
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x_ldlt[i], x_chol[i], 1e-12);

    /* Both residuals must be small */
    double r[4];
    sparse_matvec(A, x_ldlt, r);
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(Acopy);
    sparse_free(A);
}

static void test_ldlt_solve_aliased(void) {
    /* Test that x may alias b (overwrite RHS with solution). */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Solve with separate b and x first for reference */
    double b[] = {5.0, 5.0, 5.0};
    double x_ref[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x_ref), SPARSE_OK);

    /* Now solve with x aliasing b */
    double bx[] = {5.0, 5.0, 5.0};
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, bx, bx), SPARSE_OK);

    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(bx[i], x_ref[i], 1e-14);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_solve_zero_rhs(void) {
    /* A*x = 0 must give x = 0 for any nonsingular A. */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double b[] = {0.0, 0.0, 0.0};
    double x[3] = {999.0, 999.0, 999.0};
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], 0.0, 1e-14);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_solve_unfactored(void) {
    /* A zeroed struct (n == 0) is a valid empty factorization. */
    sparse_ldlt_t ldlt;
    memset(&ldlt, 0, sizeof(ldlt));
    double b = 1.0, x = 0.0;
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, &b, &x), SPARSE_OK);

    /* But n > 0 with NULL internal pointers must return BADARG. */
    ldlt.n = 1;
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, &b, &x), SPARSE_ERR_BADARG);
}

static void test_ldlt_solve_2x2_relative_residual(void) {
    /* Verify ||A*x - b|| / ||b|| < 1e-12 on a 6x6 indefinite system
     * with multiple 2x2 pivots.
     * A = [[0 2 0 1 0 0]
     *      [2 0 0 0 1 0]
     *      [0 0 0 3 0 1]
     *      [1 0 3 0 0 0]
     *      [0 1 0 0 5 1]
     *      [0 0 1 0 1 4]] */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 0, 3, 1.0);
    sparse_insert(A, 3, 0, 1.0);
    sparse_insert(A, 1, 4, 1.0);
    sparse_insert(A, 4, 1, 1.0);
    sparse_insert(A, 2, 3, 3.0);
    sparse_insert(A, 3, 2, 3.0);
    sparse_insert(A, 2, 5, 1.0);
    sparse_insert(A, 5, 2, 1.0);
    sparse_insert(A, 4, 4, 5.0);
    sparse_insert(A, 4, 5, 1.0);
    sparse_insert(A, 5, 4, 1.0);
    sparse_insert(A, 5, 5, 4.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double b[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double x[6], r[6];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);

    /* Compute ||r - b|| / ||b|| */
    double norm_res = 0.0, norm_b = 0.0;
    for (int i = 0; i < 6; i++) {
        double diff = r[i] - b[i];
        norm_res += diff * diff;
        norm_b += b[i] * b[i];
    }
    norm_res = sqrt(norm_res);
    norm_b = sqrt(norm_b);
    ASSERT_TRUE(norm_res / norm_b < 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_solve_indefinite_known_solution(void) {
    /* A = [[2,1,-1],[1,3,0],[-1,0,4]], x_true = [1,2,3]
     * b = A*x_true = [2+2-3, 1+6, -1+12] = [1,7,11]
     * Verify solve recovers x_true exactly. */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 2, -1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 0, -1.0);
    sparse_insert(A, 2, 2, 4.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double b[] = {1.0, 7.0, 11.0};
    double x[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 1.0, 1e-12);
    ASSERT_NEAR(x[1], 2.0, 1e-12);
    ASSERT_NEAR(x[2], 3.0, 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Reordering integration tests (Day 7)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: build a banded SPD matrix of size n with bandwidth bw.
 * A(i,j) = (bw+1-|i-j|) for |i-j| <= bw, giving strong diagonal dominance. */
static SparseMatrix *make_banded_spd(idx_t n, idx_t bw) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = i; j < n && j <= i + bw; j++) {
            double val = (double)(bw + 1 - (j - i));
            sparse_insert(A, i, j, val);
            if (i != j)
                sparse_insert(A, j, i, val);
        }
    return A;
}

static void test_ldlt_reorder_none_vs_amd(void) {
    /* Factor the same SPD matrix with NONE and AMD reordering.
     * Solve results must be identical (up to rounding). */
    idx_t n = 20;
    SparseMatrix *A = make_banded_spd(n, 3);

    /* Factor without reordering */
    sparse_ldlt_t ldlt_none;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt_none), SPARSE_OK);

    /* Factor with AMD */
    sparse_ldlt_opts_t opts_amd = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt_amd;
    ASSERT_ERR(sparse_ldlt_factor_opts(A, &opts_amd, &ldlt_amd), SPARSE_OK);

    /* Solve both and compare */
    double b[20], x_none[20], x_amd[20];
    for (int i = 0; i < 20; i++)
        b[i] = (double)(i + 1);

    ASSERT_ERR(sparse_ldlt_solve(&ldlt_none, b, x_none), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt_amd, b, x_amd), SPARSE_OK);

    for (int i = 0; i < 20; i++)
        ASSERT_NEAR(x_none[i], x_amd[i], 1e-10);

    /* Verify residual for AMD solve */
    double r[20];
    sparse_matvec(A, x_amd, r);
    for (int i = 0; i < 20; i++)
        ASSERT_NEAR(r[i], b[i], 1e-10);

    sparse_ldlt_free(&ldlt_none);
    sparse_ldlt_free(&ldlt_amd);
    sparse_free(A);
}

static void test_ldlt_reorder_rcm(void) {
    /* Factor with RCM reordering and verify solve result matches. */
    idx_t n = 20;
    SparseMatrix *A = make_banded_spd(n, 3);

    sparse_ldlt_t ldlt_none;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt_none), SPARSE_OK);

    sparse_ldlt_opts_t opts_rcm = {SPARSE_REORDER_RCM, 0.0};
    sparse_ldlt_t ldlt_rcm;
    ASSERT_ERR(sparse_ldlt_factor_opts(A, &opts_rcm, &ldlt_rcm), SPARSE_OK);

    double b[20], x_none[20], x_rcm[20];
    for (int i = 0; i < 20; i++)
        b[i] = 1.0;

    ASSERT_ERR(sparse_ldlt_solve(&ldlt_none, b, x_none), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt_rcm, b, x_rcm), SPARSE_OK);

    for (int i = 0; i < 20; i++)
        ASSERT_NEAR(x_none[i], x_rcm[i], 1e-10);

    sparse_ldlt_free(&ldlt_none);
    sparse_ldlt_free(&ldlt_rcm);
    sparse_free(A);
}

static void test_ldlt_reorder_fillin(void) {
    /* AMD should reduce fill-in on a structured matrix.
     * Build a 30x30 arrow matrix (dense first row/col + diagonal):
     * Without reordering: L has O(n²) fill. With AMD: much less. */
    idx_t n = 30;
    SparseMatrix *A = sparse_create(n, n);
    /* Diagonal + arrow pattern */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i > 0) {
            sparse_insert(A, i, 0, 1.0);
            sparse_insert(A, 0, i, 1.0);
        }
    }

    /* Factor without reordering */
    sparse_ldlt_t ldlt_none;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt_none), SPARSE_OK);
    idx_t nnz_none = sparse_nnz(ldlt_none.L);

    /* Factor with AMD */
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt_amd;
    ASSERT_ERR(sparse_ldlt_factor_opts(A, &opts, &ldlt_amd), SPARSE_OK);
    idx_t nnz_amd = sparse_nnz(ldlt_amd.L);

    printf("    arrow 30x30 fill-in: none=%d, AMD=%d\n", (int)nnz_none, (int)nnz_amd);

    /* AMD should produce less or equal fill-in on an arrow matrix */
    ASSERT_TRUE(nnz_amd <= nnz_none);

    /* Both must produce correct solve results */
    double b[30], x[30], r[30];
    for (int i = 0; i < 30; i++)
        b[i] = 1.0;
    ASSERT_ERR(sparse_ldlt_solve(&ldlt_amd, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 30; i++)
        ASSERT_NEAR(r[i], b[i], 1e-10);

    sparse_ldlt_free(&ldlt_none);
    sparse_ldlt_free(&ldlt_amd);
    sparse_free(A);
}

static void test_ldlt_reorder_indefinite(void) {
    /* Verify reordering works on an indefinite matrix (not just SPD).
     * 10x10 indefinite tridiagonal with AMD reordering. */
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        double diag = (i % 2 == 0) ? 4.0 : -4.0;
        sparse_insert(A, i, i, diag);
        if (i > 0) {
            sparse_insert(A, i, i - 1, 1.0);
            sparse_insert(A, i - 1, i, 1.0);
        }
    }

    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor_opts(A, &opts, &ldlt), SPARSE_OK);

    double b[10], x[10], r[10];
    for (int i = 0; i < 10; i++)
        b[i] = (double)(i + 1);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 10; i++)
        ASSERT_NEAR(r[i], b[i], 1e-10);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* Helper for SuiteSparse LDL^T validation */
static void ldlt_validate_mm(const char *path, double tol, const sparse_ldlt_opts_t *opts) {
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, path);
    if (lerr != SPARSE_OK || !A) {
        printf("    skip: %s not found\n", path);
        return;
    }

    idx_t n = sparse_rows(A);
    sparse_ldlt_t ldlt;
    sparse_err_t err;
    if (opts)
        err = sparse_ldlt_factor_opts(A, opts, &ldlt);
    else
        err = sparse_ldlt_factor(A, &ldlt);
    ASSERT_ERR(err, SPARSE_OK);

    /* Solve A*x = b where b = A*ones */
    double *ones = calloc((size_t)n, sizeof(double));
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    double *r = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);

    /* Relative residual: ||A*x - b|| / ||b|| */
    double norm_res = 0.0, norm_b = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double diff = r[i] - b[i];
        norm_res += diff * diff;
        norm_b += b[i] * b[i];
    }
    norm_res = sqrt(norm_res);
    norm_b = sqrt(norm_b);
    double relres = (norm_b > 0.0) ? norm_res / norm_b : norm_res;
    printf("    %s: n=%d, relres=%.3e\n", path, (int)n, relres);
    ASSERT_TRUE(relres < tol);

    free(ones);
    free(b);
    free(x);
    free(r);
    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_nos4(void) {
    /* nos4: 100x100 SPD matrix from structural engineering */
    ldlt_validate_mm(SS_DIR "/nos4.mtx", 1e-10, NULL);
}

static void test_ldlt_bcsstk04(void) {
    /* bcsstk04: 132x132 SPD stiffness matrix */
    ldlt_validate_mm(SS_DIR "/bcsstk04.mtx", 1e-4, NULL);
}

static void test_ldlt_nos4_amd(void) {
    /* nos4 with AMD reordering — solve result should be equally good */
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    ldlt_validate_mm(SS_DIR "/nos4.mtx", 1e-10, &opts);
}

static void test_ldlt_bcsstk04_amd(void) {
    /* bcsstk04 with AMD — compare fill-in */
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (lerr != SPARSE_OK || !A) {
        printf("    skip: bcsstk04 not found\n");
        return;
    }

    /* Without reordering */
    sparse_ldlt_t ldlt_none;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt_none), SPARSE_OK);
    idx_t nnz_none = sparse_nnz(ldlt_none.L);

    /* With AMD */
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt_amd;
    ASSERT_ERR(sparse_ldlt_factor_opts(A, &opts, &ldlt_amd), SPARSE_OK);
    idx_t nnz_amd = sparse_nnz(ldlt_amd.L);

    printf("    bcsstk04 fill-in: none=%d, AMD=%d (%.1fx)\n", (int)nnz_none, (int)nnz_amd,
           (double)nnz_none / (double)nnz_amd);

    /* Verify both solve correctly */
    idx_t n = sparse_rows(A);
    double *b = calloc((size_t)n, sizeof(double));
    double *x_none = calloc((size_t)n, sizeof(double));
    double *x_amd = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    ASSERT_ERR(sparse_ldlt_solve(&ldlt_none, b, x_none), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt_amd, b, x_amd), SPARSE_OK);

    /* Solutions should agree */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_none[i], x_amd[i], 1e-4);

    free(b);
    free(x_none);
    free(x_amd);
    sparse_ldlt_free(&ldlt_none);
    sparse_ldlt_free(&ldlt_amd);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * KKT & saddle-point tests (Day 8)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: build a KKT matrix K = [H A^T; A 0] of size (nh+nc) x (nh+nc).
 * H is nh x nh SPD (tridiagonal), A is nc x nh (sparse constraint rows). */
static SparseMatrix *make_kkt(idx_t nh, idx_t nc) {
    idx_t n = nh + nc;
    SparseMatrix *K = sparse_create(n, n);

    /* H block: tridiagonal SPD with diag=4, off=-1 */
    for (idx_t i = 0; i < nh; i++) {
        sparse_insert(K, i, i, 4.0);
        if (i > 0) {
            sparse_insert(K, i, i - 1, -1.0);
            sparse_insert(K, i - 1, i, -1.0);
        }
    }

    /* A block: each constraint row picks 2 consecutive variables */
    for (idx_t c = 0; c < nc; c++) {
        idx_t j0 = (c * 2) % nh;
        idx_t j1 = (j0 + 1) % nh;
        sparse_insert(K, nh + c, j0, 1.0);
        sparse_insert(K, j0, nh + c, 1.0);
        sparse_insert(K, nh + c, j1, 1.0);
        sparse_insert(K, j1, nh + c, 1.0);
    }
    /* Zero block at (nh:n, nh:n) is implicit (no inserts) */
    return K;
}

static void test_ldlt_kkt_small(void) {
    /* Small 6x6 KKT: H is 4x4, A is 2x4, K is 6x6 */
    SparseMatrix *K = make_kkt(4, 2);
    idx_t n = 6;

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(K, &ldlt), SPARSE_OK);

    /* Inertia: H is 4x4 SPD (4 positive), zero block is 2x2 (2 negative)
     * → inertia should be (4, 2, 0) */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos + neg + zero, n);
    ASSERT_EQ(pos, 4);
    ASSERT_EQ(neg, 2);

    /* Solve with known solution x_exact = [1,2,3,4,5,6] */
    double x_exact[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double b[6], x[6], r[6];
    sparse_matvec(K, x_exact, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);

    /* Verify ||x - x_exact|| / ||x_exact|| */
    double err_norm = 0.0, ex_norm = 0.0;
    for (int i = 0; i < 6; i++) {
        err_norm += (x[i] - x_exact[i]) * (x[i] - x_exact[i]);
        ex_norm += x_exact[i] * x_exact[i];
    }
    ASSERT_TRUE(sqrt(err_norm / ex_norm) < 1e-12);

    /* Also verify residual */
    sparse_matvec(K, x, r);
    for (int i = 0; i < 6; i++)
        ASSERT_NEAR(r[i], b[i], 1e-10);

    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

static void test_ldlt_kkt_medium(void) {
    /* Medium 20x20 KKT: H is 14x14, A is 6x14 */
    SparseMatrix *K = make_kkt(14, 6);
    idx_t n = 20;

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(K, &ldlt), SPARSE_OK);

    /* Inertia: nh positive from H, nc negative from constraints */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos + neg + zero, n);
    ASSERT_EQ(pos, 14);
    ASSERT_EQ(neg, 6);

    /* Solve A*x = b with b = K*ones */
    double ones[20], b[20], x[20], r[20];
    for (int i = 0; i < 20; i++)
        ones[i] = 1.0;
    sparse_matvec(K, ones, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);

    /* Relative residual */
    sparse_matvec(K, x, r);
    double norm_res = 0.0, norm_b = 0.0;
    for (int i = 0; i < 20; i++) {
        double d = r[i] - b[i];
        norm_res += d * d;
        norm_b += b[i] * b[i];
    }
    ASSERT_TRUE(sqrt(norm_res / norm_b) < 1e-10);

    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

static void test_ldlt_kkt_large(void) {
    /* Large 50x50 KKT: H is 35x35, A is 15x35 */
    SparseMatrix *K = make_kkt(35, 15);
    idx_t n = 50;

    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor_opts(K, &opts, &ldlt), SPARSE_OK);

    /* Inertia */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos + neg + zero, n);
    ASSERT_EQ(pos, 35);
    ASSERT_EQ(neg, 15);

    /* Solve and verify residual */
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    double *r = calloc((size_t)n, sizeof(double));
    double *ones = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(K, ones, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(K, x, r);

    double norm_res = 0.0, norm_b = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = r[i] - b[i];
        norm_res += d * d;
        norm_b += b[i] * b[i];
    }
    double relres = sqrt(norm_res / norm_b);
    printf("    KKT 50x50 (AMD): relres=%.3e\n", relres);
    ASSERT_TRUE(relres < 1e-10);

    free(b);
    free(x);
    free(r);
    free(ones);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

static void test_ldlt_saddle_point(void) {
    /* Stokes-type saddle-point matrix:
     * K = [[V  G^T]  where V = viscosity (SPD), G = gradient (divergence)
     *      [G   0 ]]
     *
     * V is 4x4 discrete Laplacian, G is 2x4 gradient operator. */
    idx_t nv = 4, np = 2, n = nv + np;
    SparseMatrix *K = sparse_create(n, n);

    /* V block: 2D discrete Laplacian on 2x2 grid */
    double V[4][4] = {{4, -1, -1, 0}, {-1, 4, 0, -1}, {-1, 0, 4, -1}, {0, -1, -1, 4}};
    for (idx_t i = 0; i < nv; i++)
        for (idx_t j = 0; j < nv; j++)
            if (V[i][j] != 0.0) {
                sparse_insert(K, i, j, V[i][j]);
            }

    /* G block: pressure gradient (2 pressure nodes, 4 velocity nodes) */
    double G[2][4] = {{1, -1, 0, 0}, {0, 0, 1, -1}};
    for (idx_t i = 0; i < np; i++)
        for (idx_t j = 0; j < nv; j++)
            if (G[i][j] != 0.0) {
                sparse_insert(K, nv + i, j, G[i][j]);
                sparse_insert(K, j, nv + i, G[i][j]);
            }

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(K, &ldlt), SPARSE_OK);

    /* Inertia: nv positive from V, np negative from zero block */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos + neg + zero, n);
    ASSERT_EQ(pos, 4);
    ASSERT_EQ(neg, 2);

    /* Log pivot strategy (BK may choose 1x1 with swaps or 2x2) */
    {
        int n2x2 = 0;
        for (idx_t i = 0; i < n; i++)
            if (ldlt.pivot_size[i] == 2)
                n2x2++;
        printf("    saddle-point 6x6: %d of %d pivots are 2x2\n", n2x2 / 2, (int)n);
    }

    /* Solve and verify residual */
    double x_exact[] = {1.0, 2.0, 3.0, 4.0, 0.5, -0.5};
    double b[6], x[6], r[6];
    sparse_matvec(K, x_exact, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(K, x, r);
    for (int i = 0; i < 6; i++)
        ASSERT_NEAR(r[i], b[i], 1e-10);

    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

static void test_ldlt_vs_lu(void) {
    /* Compare LDL^T and LU solutions on a symmetric indefinite system.
     * Both should produce the same answer (up to rounding). */
    SparseMatrix *A = sparse_create(4, 4);
    /* 4x4 symmetric indefinite */
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 0, 3, -1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, -3.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 1, 2.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 0, -1.0);
    sparse_insert(A, 3, 2, 1.0);
    sparse_insert(A, 3, 3, 5.0);

    /* LDL^T solve */
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);
    double b[] = {1.0, 2.0, 3.0, 4.0};
    double x_ldlt[4];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x_ldlt), SPARSE_OK);

    /* LU solve (needs a copy since it factors in-place) */
    SparseMatrix *Acopy = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(Acopy, SPARSE_PIVOT_PARTIAL, 1e-14), SPARSE_OK);
    double x_lu[4];
    ASSERT_ERR(sparse_lu_solve(Acopy, b, x_lu), SPARSE_OK);

    /* Solutions should agree */
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(x_ldlt[i], x_lu[i], 1e-10);

    /* Both residuals should be small */
    double r_ldlt[4], r_lu[4];
    sparse_matvec(A, x_ldlt, r_ldlt);
    sparse_matvec(A, x_lu, r_lu);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(r_ldlt[i], b[i], 1e-12);
        ASSERT_NEAR(r_lu[i], b[i], 1e-12);
    }

    sparse_ldlt_free(&ldlt);
    sparse_free(Acopy);
    sparse_free(A);
}

static void test_ldlt_kkt_vs_lu(void) {
    /* Compare LDL^T and LU on a KKT system specifically. */
    SparseMatrix *K = make_kkt(8, 4);

    /* LDL^T */
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(K, &ldlt), SPARSE_OK);

    double b[12], x_ldlt[12], x_lu[12];
    for (int i = 0; i < 12; i++)
        b[i] = (double)(i + 1);

    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x_ldlt), SPARSE_OK);

    /* LU */
    SparseMatrix *Kcopy = sparse_copy(K);
    ASSERT_ERR(sparse_lu_factor(Kcopy, SPARSE_PIVOT_PARTIAL, 1e-14), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(Kcopy, b, x_lu), SPARSE_OK);

    /* Solutions agree */
    for (int i = 0; i < 12; i++)
        ASSERT_NEAR(x_ldlt[i], x_lu[i], 1e-8);

    /* Verify LDL^T residual */
    double r[12];
    sparse_matvec(K, x_ldlt, r);
    double norm_res = 0.0, norm_b = 0.0;
    for (int i = 0; i < 12; i++) {
        double d = r[i] - b[i];
        norm_res += d * d;
        norm_b += b[i] * b[i];
    }
    ASSERT_TRUE(sqrt(norm_res / norm_b) < 1e-10);

    sparse_ldlt_free(&ldlt);
    sparse_free(Kcopy);
    sparse_free(K);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Scaled tolerance & eigenvalue tests (Day 9)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: build a scaled indefinite matrix.  Takes a base matrix pattern
 * and scales all entries by s.  Base: [[2,1,-1],[1,3,0],[-1,0,4]] */
static SparseMatrix *make_scaled_indefinite_3x3(double s) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0 * s);
    sparse_insert(A, 0, 1, 1.0 * s);
    sparse_insert(A, 0, 2, -1.0 * s);
    sparse_insert(A, 1, 0, 1.0 * s);
    sparse_insert(A, 1, 1, 3.0 * s);
    sparse_insert(A, 2, 0, -1.0 * s);
    sparse_insert(A, 2, 2, 4.0 * s);
    return A;
}

static void test_ldlt_scale_tiny(void) {
    /* Factor and solve at scale 1e-35: verify norm-relative tolerance
     * prevents false singularity. */
    double s = 1e-35;
    SparseMatrix *A = make_scaled_indefinite_3x3(s);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Solve A*x = b where b = A*[1,2,3] */
    double x_exact[] = {1.0, 2.0, 3.0};
    double b[3], x[3];
    sparse_matvec(A, x_exact, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);

    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-8);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_scale_huge(void) {
    /* Factor and solve at scale 1e+35: no false singularity. */
    double s = 1e+35;
    SparseMatrix *A = make_scaled_indefinite_3x3(s);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double x_exact[] = {1.0, 2.0, 3.0};
    double b[3], x[3];
    sparse_matvec(A, x_exact, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);

    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-8);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_scale_tiny_2x2(void) {
    /* Scaled 2x2 indefinite: [[0, s], [s, 0]] at s = 1e-30 */
    double s = 1e-30;
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 1, s);
    sparse_insert(A, 1, 0, s);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);
    ASSERT_EQ(ldlt.pivot_size[0], 2);

    double b[] = {s, 0.0};
    double x[2];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 0.0, 1e-8);
    ASSERT_NEAR(x[1], 1.0, 1e-8);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_known_eigenvalues(void) {
    /* Diagonal matrix with known eigenvalues: diag(3, 1, -1, -2).
     * D should capture these exactly. Inertia = (2, 2, 0). */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    double eigs[] = {3.0, 1.0, -1.0, -2.0};
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, eigs[i]);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* D should equal the eigenvalues (diagonal → no fill, L = I) */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_EQ(ldlt.pivot_size[i], 1);
        ASSERT_NEAR(ldlt.D[i], eigs[i], 1e-14);
    }

    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 2);
    ASSERT_EQ(neg, 2);
    ASSERT_EQ(zero, 0);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_negative_definite(void) {
    /* Negative-definite 3x3: A = -I * scale.  Inertia = (0, 3, 0). */
    idx_t n = 3;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, -5.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 0);
    ASSERT_EQ(neg, 3);
    ASSERT_EQ(zero, 0);

    /* Solve (-5I)*x = [-10,-15,-20] → x = [2,3,4] */
    double b[] = {-10.0, -15.0, -20.0};
    double x[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_NEAR(x[0], 2.0, 1e-14);
    ASSERT_NEAR(x[1], 3.0, 1e-14);
    ASSERT_NEAR(x[2], 4.0, 1e-14);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_eigenvalue_spread(void) {
    /* Matrix with eigenvalues spanning many magnitudes.
     * Diagonal: [1e6, 1e3, 1, 1e-3, -1e-3, -1, -1e3, -1e6]
     * This tests that BK handles widely varying pivot magnitudes. */
    idx_t n = 8;
    double eigs[] = {1e6, 1e3, 1.0, 1e-3, -1e-3, -1.0, -1e3, -1e6};
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, eigs[i]);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Inertia: 4 positive, 4 negative */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 4);
    ASSERT_EQ(neg, 4);

    /* D should capture eigenvalues */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(ldlt.D[i], eigs[i], fabs(eigs[i]) * 1e-14);

    /* Solve and verify */
    double b[8], x[8];
    double x_exact[] = {1, 2, 3, 4, 5, 6, 7, 8};
    sparse_matvec(A, x_exact, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    for (int i = 0; i < 8; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-8);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_rotated_indefinite(void) {
    /* Non-diagonal indefinite with known inertia.
     * Build A = Q * diag(5, 2, -1, -3) * Q^T where Q is a rotation.
     * Use a simple rotation: Givens in the (0,1) plane by 45 degrees.
     *
     * Q = [[c -s 0 0], [s c 0 0], [0 0 1 0], [0 0 0 1]]
     * with c = s = 1/sqrt(2).
     *
     * A = Q * D * Q^T where D = diag(5, 2, -1, -3)
     * A(0,0) = c^2*5 + s^2*2 = 0.5*5 + 0.5*2 = 3.5
     * A(0,1) = c*s*5 - c*s*2 = 0.5*(5-2)*1 = 1.5  (using c*s = 0.5)
     * A(1,0) = 1.5
     * A(1,1) = s^2*5 + c^2*2 = 3.5
     * A(2,2) = -1, A(3,3) = -3
     */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 3.5);
    sparse_insert(A, 0, 1, 1.5);
    sparse_insert(A, 1, 0, 1.5);
    sparse_insert(A, 1, 1, 3.5);
    sparse_insert(A, 2, 2, -1.0);
    sparse_insert(A, 3, 3, -3.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Inertia must be (2, 2, 0) — matches the 4 eigenvalues */
    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 2);
    ASSERT_EQ(neg, 2);
    ASSERT_EQ(zero, 0);

    /* Solve and verify residual */
    double b[] = {1.0, 2.0, 3.0, 4.0};
    double x[4], r[4];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(A, x, r);
    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(r[i], b[i], 1e-12);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_inertia_2x2_block(void) {
    /* Verify inertia computation for 2x2 D blocks.
     * A = [[0,5],[5,0]] → 2x2 block with det = -25 < 0 → (1+, 1-)
     * Eigenvalues are +5 and -5. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 1, 5.0);
    sparse_insert(A, 1, 0, 5.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);
    ASSERT_EQ(ldlt.pivot_size[0], 2);

    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 1);
    ASSERT_EQ(neg, 1);
    ASSERT_EQ(zero, 0);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Iterative refinement & condition estimation tests (Day 10)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ldlt_refine_null(void) {
    ASSERT_ERR(sparse_ldlt_refine(NULL, NULL, NULL, NULL, 0, 0.0), SPARSE_ERR_NULL);
}

static void test_ldlt_refine_basic(void) {
    /* Basic refinement: should not degrade a good solution. */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double b[] = {5.0, 5.0, 5.0};
    double x[3];
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_refine(A, &ldlt, b, x, 3, 1e-15), SPARSE_OK);

    /* Verify solution is still good */
    double r[3];
    sparse_matvec(A, x, r);
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(r[i], b[i], 1e-13);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_refine_improves(void) {
    /* Start with a deliberately perturbed solution and verify refinement
     * improves it. Use an ill-conditioned system where initial solve
     * has noticeable error. */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    /* Hilbert-like symmetric indefinite: A(i,j) = (-1)^(i+j) / (i+j+1) + 10*I */
    for (idx_t i = 0; i < n; i++)
        for (idx_t j = 0; j < n; j++) {
            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            double val = sign / (double)(i + j + 1);
            if (i == j)
                val += 10.0;
            sparse_insert(A, i, j, val);
        }

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    /* Solve with known answer */
    double x_exact[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[5], x[5], r[5];
    sparse_matvec(A, x_exact, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);

    /* Perturb the solution slightly */
    for (int i = 0; i < 5; i++)
        x[i] += 1e-8;

    /* Compute pre-refinement residual */
    sparse_matvec(A, x, r);
    double pre_res = 0.0;
    for (int i = 0; i < 5; i++)
        pre_res += (r[i] - b[i]) * (r[i] - b[i]);
    pre_res = sqrt(pre_res);

    /* Refine */
    ASSERT_ERR(sparse_ldlt_refine(A, &ldlt, b, x, 5, 1e-15), SPARSE_OK);

    /* Compute post-refinement residual */
    sparse_matvec(A, x, r);
    double post_res = 0.0;
    for (int i = 0; i < 5; i++)
        post_res += (r[i] - b[i]) * (r[i] - b[i]);
    post_res = sqrt(post_res);

    printf("    refine: pre=%.3e, post=%.3e\n", pre_res, post_res);
    ASSERT_TRUE(post_res <= pre_res);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_refine_zero_rhs(void) {
    /* Refinement with b=0 should keep x=0. */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 1, 5.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double b[] = {0.0, 0.0};
    double x[] = {0.0, 0.0};
    ASSERT_ERR(sparse_ldlt_refine(A, &ldlt, b, x, 3, 1e-15), SPARSE_OK);
    ASSERT_NEAR(x[0], 0.0, 1e-15);
    ASSERT_NEAR(x[1], 0.0, 1e-15);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_condest_null(void) {
    ASSERT_ERR(sparse_ldlt_condest(NULL, NULL, NULL), SPARSE_ERR_NULL);
}

static void test_ldlt_condest_identity(void) {
    /* Condition number of identity matrix should be 1. */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double cond;
    ASSERT_ERR(sparse_ldlt_condest(A, &ldlt, &cond), SPARSE_OK);
    printf("    condest(I): %.3f\n", cond);
    ASSERT_NEAR(cond, 1.0, 0.1);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_condest_wellcond(void) {
    /* Well-conditioned SPD: diag(2,3,4,5) → cond_1 = 5/2 = 2.5 */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 3, 3, 5.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double cond;
    ASSERT_ERR(sparse_ldlt_condest(A, &ldlt, &cond), SPARSE_OK);
    printf("    condest(diag): %.3f (true=2.5)\n", cond);
    /* Should be within 10x of true value */
    ASSERT_TRUE(cond >= 1.0 && cond <= 25.0);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_condest_illcond(void) {
    /* Ill-conditioned: diag(1, 1e-8) → cond_1 = 1e8 */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1e-8);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double cond;
    ASSERT_ERR(sparse_ldlt_condest(A, &ldlt, &cond), SPARSE_OK);
    printf("    condest(ill): %.3e (true=1e8)\n", cond);
    /* Should detect large condition number (within 10x) */
    ASSERT_TRUE(cond >= 1e7);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

static void test_ldlt_condest_indefinite(void) {
    /* Indefinite system condition estimation.
     * A = [[2,1],[-1,2]] is NOT symmetric — use [[2,1],[1,-2]] instead.
     * Eigenvalues of [[2,1],[1,-2]]: sqrt(5) ≈ 2.236 and -sqrt(5).
     * cond_1 = (||A||_1 * ||A^{-1}||_1). */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, -2.0);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);

    double cond;
    ASSERT_ERR(sparse_ldlt_condest(A, &ldlt, &cond), SPARSE_OK);
    printf("    condest(indef): %.3f\n", cond);
    /* Should be a finite positive number */
    ASSERT_TRUE(cond > 0.0 && cond < 1e10);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SuiteSparse validation & performance tests (Day 12)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ldlt_bcsstk04_vs_cholesky(void) {
    /* bcsstk04 is SPD — LDL^T and Cholesky must produce the same solution. */
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (lerr != SPARSE_OK || !A) {
        printf("    skip: bcsstk04 not found\n");
        return;
    }

    idx_t n = sparse_rows(A);
    double *b = calloc((size_t)n, sizeof(double));
    double *x_ldlt = calloc((size_t)n, sizeof(double));
    double *x_chol = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    /* LDL^T solve */
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x_ldlt), SPARSE_OK);

    /* Cholesky solve */
    SparseMatrix *Acopy = sparse_copy(A);
    ASSERT_ERR(sparse_cholesky_factor(Acopy), SPARSE_OK);
    ASSERT_ERR(sparse_cholesky_solve(Acopy, b, x_chol), SPARSE_OK);

    /* Solutions must agree */
    double max_diff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(x_ldlt[i] - x_chol[i]);
        if (d > max_diff)
            max_diff = d;
    }
    printf("    bcsstk04 LDL^T vs Cholesky: max|diff| = %.3e\n", max_diff);
    ASSERT_TRUE(max_diff < 1e-4);

    free(b);
    free(x_ldlt);
    free(x_chol);
    sparse_ldlt_free(&ldlt);
    sparse_free(Acopy);
    sparse_free(A);
}

static void test_ldlt_kkt_100(void) {
    /* Synthetic 100x100 KKT: H is 70x70, A is 30x70. */
    SparseMatrix *K = make_kkt(70, 30);
    idx_t n = 100;

    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor_opts(K, &opts, &ldlt), SPARSE_OK);

    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 70);
    ASSERT_EQ(neg, 30);

    double *ones = calloc((size_t)n, sizeof(double));
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    double *r = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(K, ones, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(K, x, r);

    double norm_res = 0.0, norm_b = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = r[i] - b[i];
        norm_res += d * d;
        norm_b += b[i] * b[i];
    }
    double relres = sqrt(norm_res / norm_b);
    printf("    KKT 100x100: relres=%.3e, nnz(L)=%d\n", relres, (int)sparse_nnz(ldlt.L));
    ASSERT_TRUE(relres < 1e-10);

    free(ones);
    free(b);
    free(x);
    free(r);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

static void test_ldlt_kkt_500(void) {
    /* Synthetic 500x500 KKT: H is 350x350, A is 150x350. */
    SparseMatrix *K = make_kkt(350, 150);
    idx_t n = 500;

    sparse_ldlt_opts_t opts = {SPARSE_REORDER_AMD, 0.0};
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor_opts(K, &opts, &ldlt), SPARSE_OK);

    idx_t pos, neg, zero;
    ASSERT_ERR(sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero), SPARSE_OK);
    ASSERT_EQ(pos, 350);
    ASSERT_EQ(neg, 150);

    double *ones = calloc((size_t)n, sizeof(double));
    double *b = calloc((size_t)n, sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    double *r = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(K, ones, b);
    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x), SPARSE_OK);
    sparse_matvec(K, x, r);

    double norm_res = 0.0, norm_b = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = r[i] - b[i];
        norm_res += d * d;
        norm_b += b[i] * b[i];
    }
    double relres = sqrt(norm_res / norm_b);
    printf("    KKT 500x500: relres=%.3e, nnz(L)=%d\n", relres, (int)sparse_nnz(ldlt.L));
    ASSERT_TRUE(relres < 1e-10);

    free(ones);
    free(b);
    free(x);
    free(r);
    sparse_ldlt_free(&ldlt);
    sparse_free(K);
}

static void test_ldlt_vs_lu_fillin(void) {
    /* Compare fill-in: LDL^T vs LU on symmetric matrices.
     * LDL^T exploits symmetry → should have roughly half the fill. */
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (lerr != SPARSE_OK || !A) {
        printf("    skip: nos4 not found\n");
        return;
    }

    idx_t n = sparse_rows(A);

    /* LDL^T */
    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);
    idx_t nnz_ldlt = sparse_nnz(ldlt.L);

    /* LU (needs copy — factors in-place) */
    SparseMatrix *Acopy = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(Acopy, SPARSE_PIVOT_PARTIAL, 1e-14), SPARSE_OK);
    idx_t nnz_lu = sparse_nnz(Acopy);

    printf("    nos4 (n=%d): nnz(L_ldlt)=%d, nnz(LU)=%d, ratio=%.2f\n", (int)n, (int)nnz_ldlt,
           (int)nnz_lu, (double)nnz_ldlt / (double)nnz_lu);

    /* LDL^T fill-in should be <= LU fill-in */
    ASSERT_TRUE(nnz_ldlt <= nnz_lu);

    /* Both must solve correctly */
    double *b = calloc((size_t)n, sizeof(double));
    double *x_ldlt = calloc((size_t)n, sizeof(double));
    double *x_lu = calloc((size_t)n, sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    ASSERT_ERR(sparse_ldlt_solve(&ldlt, b, x_ldlt), SPARSE_OK);
    ASSERT_ERR(sparse_lu_solve(Acopy, b, x_lu), SPARSE_OK);

    /* Solutions should agree */
    double max_diff = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(x_ldlt[i] - x_lu[i]);
        if (d > max_diff)
            max_diff = d;
    }
    printf("    nos4 LDL^T vs LU: max|diff| = %.3e\n", max_diff);
    ASSERT_TRUE(max_diff < 1e-8);

    free(b);
    free(x_ldlt);
    free(x_lu);
    sparse_ldlt_free(&ldlt);
    sparse_free(Acopy);
    sparse_free(A);
}

static void test_ldlt_vs_lu_fillin_bcsstk04(void) {
    /* Same fill-in comparison on bcsstk04 */
    SparseMatrix *A = NULL;
    sparse_err_t lerr = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (lerr != SPARSE_OK || !A) {
        printf("    skip: bcsstk04 not found\n");
        return;
    }

    idx_t n = sparse_rows(A);

    sparse_ldlt_t ldlt;
    ASSERT_ERR(sparse_ldlt_factor(A, &ldlt), SPARSE_OK);
    idx_t nnz_ldlt = sparse_nnz(ldlt.L);

    SparseMatrix *Acopy = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(Acopy, SPARSE_PIVOT_PARTIAL, 1e-14), SPARSE_OK);
    idx_t nnz_lu = sparse_nnz(Acopy);

    printf("    bcsstk04 (n=%d): nnz(L_ldlt)=%d, nnz(LU)=%d, ratio=%.2f\n", (int)n, (int)nnz_ldlt,
           (int)nnz_lu, (double)nnz_ldlt / (double)nnz_lu);

    ASSERT_TRUE(nnz_ldlt <= nnz_lu);

    sparse_ldlt_free(&ldlt);
    sparse_free(Acopy);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("LDL^T Factorization Tests");

    /* Entry validation */
    RUN_TEST(test_ldlt_null_args);
    RUN_TEST(test_ldlt_non_square);
    RUN_TEST(test_ldlt_non_symmetric);
    RUN_TEST(test_ldlt_empty_matrix);
    RUN_TEST(test_ldlt_1x1);
    RUN_TEST(test_ldlt_1x1_negative);

    /* Diagonal matrices */
    RUN_TEST(test_ldlt_diagonal_spd);
    RUN_TEST(test_ldlt_diagonal_indefinite);
    RUN_TEST(test_ldlt_singular_diagonal);

    /* 1x1 pivot factorization (Day 3) */
    RUN_TEST(test_ldlt_spd_3x3);
    RUN_TEST(test_ldlt_spd_5x5_tridiag);
    RUN_TEST(test_ldlt_identity);
    RUN_TEST(test_ldlt_reconstruct_ldlt);
    RUN_TEST(test_ldlt_solve_multiple_rhs);

    /* 2×2 pivot tests (Day 4) */
    RUN_TEST(test_ldlt_2x2_indefinite);
    RUN_TEST(test_ldlt_2x2_with_trailing);
    RUN_TEST(test_ldlt_2x2_reconstruct);
    RUN_TEST(test_ldlt_mixed_pivots_4x4);
    RUN_TEST(test_ldlt_2x2_bk_swap);
    RUN_TEST(test_ldlt_2x2_kkt_like);

    /* Hardening & edge cases (Day 5) */
    RUN_TEST(test_ldlt_singular_zero_block);
    RUN_TEST(test_ldlt_singular_rank_deficient);
    RUN_TEST(test_ldlt_multiple_2x2_blocks);
    RUN_TEST(test_ldlt_large_indefinite);
    RUN_TEST(test_ldlt_tridiag_indefinite);
    RUN_TEST(test_ldlt_error_recovery);
    RUN_TEST(test_ldlt_2x2_with_reorder);
    RUN_TEST(test_ldlt_scaled_matrix);

    /* Solve tests (Day 6) */
    RUN_TEST(test_ldlt_solve_vs_cholesky);
    RUN_TEST(test_ldlt_solve_aliased);
    RUN_TEST(test_ldlt_solve_zero_rhs);
    RUN_TEST(test_ldlt_solve_unfactored);
    RUN_TEST(test_ldlt_solve_2x2_relative_residual);
    RUN_TEST(test_ldlt_solve_indefinite_known_solution);

    /* Reordering integration (Day 7) */
    RUN_TEST(test_ldlt_reorder_none_vs_amd);
    RUN_TEST(test_ldlt_reorder_rcm);
    RUN_TEST(test_ldlt_reorder_fillin);
    RUN_TEST(test_ldlt_reorder_indefinite);
    RUN_TEST(test_ldlt_nos4);
    RUN_TEST(test_ldlt_bcsstk04);
    RUN_TEST(test_ldlt_nos4_amd);
    RUN_TEST(test_ldlt_bcsstk04_amd);

    /* KKT & saddle-point tests (Day 8) */
    RUN_TEST(test_ldlt_kkt_small);
    RUN_TEST(test_ldlt_kkt_medium);
    RUN_TEST(test_ldlt_kkt_large);
    RUN_TEST(test_ldlt_saddle_point);
    RUN_TEST(test_ldlt_vs_lu);
    RUN_TEST(test_ldlt_kkt_vs_lu);

    /* Scaled tolerance & eigenvalue tests (Day 9) */
    RUN_TEST(test_ldlt_scale_tiny);
    RUN_TEST(test_ldlt_scale_huge);
    RUN_TEST(test_ldlt_scale_tiny_2x2);
    RUN_TEST(test_ldlt_known_eigenvalues);
    RUN_TEST(test_ldlt_negative_definite);
    RUN_TEST(test_ldlt_eigenvalue_spread);
    RUN_TEST(test_ldlt_rotated_indefinite);
    RUN_TEST(test_ldlt_inertia_2x2_block);

    /* Iterative refinement & condition estimation (Day 10) */
    RUN_TEST(test_ldlt_refine_null);
    RUN_TEST(test_ldlt_refine_basic);
    RUN_TEST(test_ldlt_refine_improves);
    RUN_TEST(test_ldlt_refine_zero_rhs);
    RUN_TEST(test_ldlt_condest_null);
    RUN_TEST(test_ldlt_condest_identity);
    RUN_TEST(test_ldlt_condest_wellcond);
    RUN_TEST(test_ldlt_condest_illcond);
    RUN_TEST(test_ldlt_condest_indefinite);

    /* SuiteSparse validation & performance (Day 12) */
    RUN_TEST(test_ldlt_bcsstk04_vs_cholesky);
    RUN_TEST(test_ldlt_kkt_100);
    RUN_TEST(test_ldlt_kkt_500);
    RUN_TEST(test_ldlt_vs_lu_fillin);
    RUN_TEST(test_ldlt_vs_lu_fillin_bcsstk04);

    /* Free/cleanup */
    RUN_TEST(test_ldlt_free_zeroed);
    RUN_TEST(test_ldlt_inertia_null);

    TEST_SUITE_END();
}
