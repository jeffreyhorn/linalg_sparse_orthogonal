#include "sparse_lu.h"
#include "sparse_matrix.h"
#include "sparse_reorder.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_permute tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Identity permutation → output equals input */
static void test_permute_identity(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    idx_t perm[] = {0, 1, 2};
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    ASSERT_EQ(sparse_nnz(B), sparse_nnz(A));
    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 1.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 0, 2), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 5.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 0), 7.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 2), 9.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

/* Reverse permutation on diagonal → diagonal reversed */
static void test_permute_reverse_diagonal(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);

    idx_t perm[] = {2, 1, 0};
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    /* B(0,0) = A(2,2) = 3, B(1,1) = A(1,1) = 2, B(2,2) = A(0,0) = 1 */
    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 2, 2), 1.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

/* Known permutation on small matrix → verify entries */
static void test_permute_known(void) {
    /* A = [1 2; 3 4], perm = [1, 0] (swap rows and cols)
     * B(0,0) = A(1,1) = 4, B(0,1) = A(1,0) = 3
     * B(1,0) = A(0,1) = 2, B(1,1) = A(0,0) = 1 */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 3.0);
    sparse_insert(A, 1, 1, 4.0);

    idx_t perm[] = {1, 0};
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);

    ASSERT_NEAR(sparse_get_phys(B, 0, 0), 4.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 0, 1), 3.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 0), 2.0, 0.0);
    ASSERT_NEAR(sparse_get_phys(B, 1, 1), 1.0, 0.0);

    sparse_free(A);
    sparse_free(B);
}

/* Permute preserves nnz */
static void test_permute_preserves_nnz(void) {
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 2, 2.0);
    sparse_insert(A, 2, 3, 3.0);
    sparse_insert(A, 3, 0, 4.0);

    idx_t perm[] = {3, 0, 1, 2};
    SparseMatrix *B = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &B), SPARSE_OK);
    ASSERT_NOT_NULL(B);
    ASSERT_EQ(sparse_nnz(B), 4);

    sparse_free(A);
    sparse_free(B);
}

/* NULL inputs → proper error codes */
static void test_permute_null(void) {
    idx_t perm[] = {0};
    SparseMatrix *B;
    SparseMatrix *A = sparse_create(1, 1);
    ASSERT_ERR(sparse_permute(NULL, perm, perm, &B), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_permute(A, NULL, perm, &B), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_permute(A, perm, NULL, &B), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_permute(A, perm, perm, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_bandwidth tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_bandwidth_diagonal(void) {
    SparseMatrix *A = sparse_create(5, 5);
    for (idx_t i = 0; i < 5; i++)
        sparse_insert(A, i, i, 1.0);
    ASSERT_EQ(sparse_bandwidth(A), 0);
    sparse_free(A);
}

static void test_bandwidth_tridiag(void) {
    SparseMatrix *A = sparse_create(5, 5);
    for (idx_t i = 0; i < 5; i++) {
        sparse_insert(A, i, i, 2.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < 4)
            sparse_insert(A, i, i + 1, -1.0);
    }
    ASSERT_EQ(sparse_bandwidth(A), 1);
    sparse_free(A);
}

static void test_bandwidth_full(void) {
    /* Entry at (0, 4) gives bandwidth 4 */
    SparseMatrix *A = sparse_create(5, 5);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 4, 1.0);
    sparse_insert(A, 4, 4, 1.0);
    ASSERT_EQ(sparse_bandwidth(A), 4);
    sparse_free(A);
}

static void test_bandwidth_null(void) { ASSERT_EQ(sparse_bandwidth(NULL), 0); }

/* ═══════════════════════════════════════════════════════════════════════
 * Permute + solve integration test
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_permute_then_solve(void) {
    /* A = [4 -1 0; -1 4 -1; 0 -1 4], solve A*x = b */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, -1.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, -1.0);
    sparse_insert(A, 2, 1, -1.0);
    sparse_insert(A, 2, 2, 4.0);

    double b_orig[] = {1.0, 2.0, 3.0};

    /* Permute with perm = [2, 0, 1] */
    idx_t perm[] = {2, 0, 1};
    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);

    /* Permute the RHS: pb[i] = b[perm[i]] */
    double pb[3];
    for (int i = 0; i < 3; i++)
        pb[i] = b_orig[perm[i]];

    /* Factor and solve the permuted system */
    ASSERT_ERR(sparse_lu_factor(PA, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    double xp[3];
    ASSERT_ERR(sparse_lu_solve(PA, pb, xp), SPARSE_OK);

    /* Unpermute: x_orig[perm[i]] = xp[i] */
    double x_orig[3];
    for (int i = 0; i < 3; i++)
        x_orig[perm[i]] = xp[i];

    /* Verify A * x_orig ≈ b_orig */
    double r[3];
    sparse_matvec(A, x_orig, r);
    for (int i = 0; i < 3; i++)
        r[i] -= b_orig[i];
    double rnorm = 0.0;
    for (int i = 0; i < 3; i++) {
        double a = fabs(r[i]);
        if (a > rnorm)
            rnorm = a;
    }
    ASSERT_TRUE(rnorm < 1e-12);

    sparse_free(A);
    sparse_free(PA);
}

/* ═══════════════════════════════════════════════════════════════════════
 * RCM tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Check that perm is a valid permutation of 0..n-1 */
static int is_valid_perm(const idx_t *perm, idx_t n) {
    int *seen = calloc((size_t)n, sizeof(int));
    if (!seen)
        return 0;
    for (idx_t i = 0; i < n; i++) {
        if (perm[i] < 0 || perm[i] >= n || seen[perm[i]]) {
            free(seen);
            return 0;
        }
        seen[perm[i]] = 1;
    }
    free(seen);
    return 1;
}

/* Arrow matrix: dense first row/column → RCM should reduce bandwidth */
static void test_rcm_arrow(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    /* Arrow: row 0 connects to all, col 0 connects to all, diagonal */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0);
        if (i > 0) {
            sparse_insert(A, 0, i, 1.0);
            sparse_insert(A, i, 0, 1.0);
        }
    }

    idx_t bw_before = sparse_bandwidth(A);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_NOT_NULL(perm);
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);
    idx_t bw_after = sparse_bandwidth(PA);

    /* RCM should not increase bandwidth (may reduce or keep same) */
    ASSERT_TRUE(bw_after <= bw_before);

    free(perm);
    sparse_free(A);
    sparse_free(PA);
}

/* Already-banded (tridiagonal) → RCM should not worsen */
static void test_rcm_tridiag(void) {
    idx_t n = 20;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0)
            sparse_insert(A, i, i - 1, -1.0);
        if (i < n - 1)
            sparse_insert(A, i, i + 1, -1.0);
    }

    idx_t bw_before = sparse_bandwidth(A);
    ASSERT_EQ(bw_before, 1);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_NOT_NULL(perm);
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);
    idx_t bw_after = sparse_bandwidth(PA);
    ASSERT_TRUE(bw_after <= bw_before);

    free(perm);
    sparse_free(A);
    sparse_free(PA);
}

/* Diagonal matrix → any permutation is valid */
static void test_rcm_diagonal(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    /* Bandwidth should remain 0 */
    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);
    ASSERT_EQ(sparse_bandwidth(PA), 0);

    free(perm);
    sparse_free(A);
    sparse_free(PA);
}

/* Block diagonal (disconnected graph) → should handle without errors */
static void test_rcm_disconnected(void) {
    /* Two 3x3 blocks, no connections between them */
    idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    /* Block 1: rows 0-2 */
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);
    sparse_insert(A, 2, 2, 4.0);
    /* Block 2: rows 3-5 */
    sparse_insert(A, 3, 3, 4.0);
    sparse_insert(A, 3, 4, 1.0);
    sparse_insert(A, 4, 3, 1.0);
    sparse_insert(A, 4, 4, 4.0);
    sparse_insert(A, 4, 5, 1.0);
    sparse_insert(A, 5, 4, 1.0);
    sparse_insert(A, 5, 5, 4.0);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    free(perm);
    sparse_free(A);
}

/* RCM + factor + solve produces correct result */
static void test_rcm_solve(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    /* Arrow-like: node 0 connects to all, plus diagonal */
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i > 0) {
            sparse_insert(A, 0, i, 1.0);
            sparse_insert(A, i, 0, 1.0);
        }
    }

    /* RCM reorder */
    idx_t *p = malloc((size_t)n * sizeof(idx_t));
    ASSERT_ERR(sparse_reorder_rcm(A, p), SPARSE_OK);

    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, p, p, &PA), SPARSE_OK);

    /* RHS: b = A * ones */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* Permute RHS */
    double *pb = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        pb[i] = b[p[i]];

    /* Factor and solve permuted system */
    ASSERT_ERR(sparse_lu_factor(PA, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    double *xp = malloc((size_t)n * sizeof(double));
    ASSERT_ERR(sparse_lu_solve(PA, pb, xp), SPARSE_OK);

    /* Unpermute solution */
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x[p[i]] = xp[i];

    /* Verify residual */
    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(A, x, r);
    double rnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        r[i] -= b[i];
        double a = fabs(r[i]);
        if (a > rnorm)
            rnorm = a;
    }
    ASSERT_TRUE(rnorm < 1e-10);

    free(p);
    free(ones);
    free(b);
    free(pb);
    free(xp);
    free(x);
    free(r);
    sparse_free(A);
    sparse_free(PA);
}

/* ═══════════════════════════════════════════════════════════════════════
 * RCM on SuiteSparse matrices
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* Helper: load matrix, RCM reorder, factor, solve, check residual.
 * Also reports bandwidth before/after and fill-in comparison. */
static void rcm_validate_matrix(const char *path, double res_tol) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, path), SPARSE_OK);
    idx_t n = sparse_rows(A);

    /* Bandwidth before */
    idx_t bw_before = sparse_bandwidth(A);

    /* RCM */
    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_NOT_NULL(perm);
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);

    idx_t bw_after = sparse_bandwidth(PA);

    /* Factor original (for fill-in comparison) */
    SparseMatrix *LU_orig = sparse_copy(A);
    ASSERT_NOT_NULL(LU_orig);
    ASSERT_ERR(sparse_lu_factor(LU_orig, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    idx_t fill_orig = sparse_nnz(LU_orig);

    /* Factor reordered */
    SparseMatrix *LU_rcm = sparse_copy(PA);
    ASSERT_NOT_NULL(LU_rcm);
    ASSERT_ERR(sparse_lu_factor(LU_rcm, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    idx_t fill_rcm = sparse_nnz(LU_rcm);

    /* Solve with RCM-reordered system */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *pb = malloc((size_t)n * sizeof(double));
    double *xp = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(pb);
    ASSERT_NOT_NULL(xp);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(r);
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    /* Permute RHS */
    for (idx_t i = 0; i < n; i++)
        pb[i] = b[perm[i]];

    ASSERT_ERR(sparse_lu_solve(LU_rcm, pb, xp), SPARSE_OK);

    /* Unpermute solution */
    for (idx_t i = 0; i < n; i++)
        x[perm[i]] = xp[i];

    /* Residual */
    sparse_matvec(A, x, r);
    double rnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        r[i] -= b[i];
        double a = fabs(r[i]);
        if (a > rnorm)
            rnorm = a;
    }

    printf("    %s: bw %d->%d, fill %d->%d (%.1fx), res=%.2e\n", path, (int)bw_before,
           (int)bw_after, (int)fill_orig, (int)fill_rcm,
           fill_orig > 0 ? (double)fill_rcm / (double)fill_orig : 0.0, rnorm);

    ASSERT_TRUE(rnorm < res_tol);

    free(perm);
    free(ones);
    free(b);
    free(pb);
    free(xp);
    free(x);
    free(r);
    sparse_free(A);
    sparse_free(PA);
    sparse_free(LU_orig);
    sparse_free(LU_rcm);
}

static void test_rcm_west0067(void) { rcm_validate_matrix(SS_DIR "/west0067.mtx", 1e-8); }

static void test_rcm_nos4(void) { rcm_validate_matrix(SS_DIR "/nos4.mtx", 1e-8); }

static void test_rcm_bcsstk04(void) { rcm_validate_matrix(SS_DIR "/bcsstk04.mtx", 1e-4); }

static void test_rcm_steam1(void) { rcm_validate_matrix(SS_DIR "/steam1.mtx", 1e-2); }

/* ═══════════════════════════════════════════════════════════════════════
 * AMD tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* AMD on small matrix: valid permutation and fill-in not worse */
static void test_amd_small(void) {
    /* 5x5 arrow matrix */
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 5.0);
        if (i > 0) {
            sparse_insert(A, 0, i, 1.0);
            sparse_insert(A, i, 0, 1.0);
        }
    }

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_NOT_NULL(perm);
    ASSERT_ERR(sparse_reorder_amd(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    /* Factor original vs AMD-reordered, compare fill-in */
    SparseMatrix *LU_orig = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU_orig, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    idx_t fill_orig = sparse_nnz(LU_orig);

    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, perm, perm, &PA), SPARSE_OK);
    SparseMatrix *LU_amd = sparse_copy(PA);
    ASSERT_ERR(sparse_lu_factor(LU_amd, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    idx_t fill_amd = sparse_nnz(LU_amd);

    /* AMD should not increase fill-in vs natural on arrow matrix */
    ASSERT_TRUE(fill_amd <= fill_orig);

    free(perm);
    sparse_free(A);
    sparse_free(PA);
    sparse_free(LU_orig);
    sparse_free(LU_amd);
}

/* AMD on diagonal: valid permutation */
static void test_amd_diagonal(void) {
    idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_NOT_NULL(perm);
    ASSERT_ERR(sparse_reorder_amd(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    free(perm);
    sparse_free(A);
}

/* AMD + factor + solve produces correct result */
static void test_amd_solve(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i > 0) {
            sparse_insert(A, 0, i, 1.0);
            sparse_insert(A, i, 0, 1.0);
        }
    }

    idx_t *p = malloc((size_t)n * sizeof(idx_t));
    ASSERT_ERR(sparse_reorder_amd(A, p), SPARSE_OK);

    SparseMatrix *PA = NULL;
    ASSERT_ERR(sparse_permute(A, p, p, &PA), SPARSE_OK);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    double *pb = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        pb[i] = b[p[i]];

    ASSERT_ERR(sparse_lu_factor(PA, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    double *xp = malloc((size_t)n * sizeof(double));
    ASSERT_ERR(sparse_lu_solve(PA, pb, xp), SPARSE_OK);

    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x[p[i]] = xp[i];

    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(A, x, r);
    double rnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        r[i] -= b[i];
        double a = fabs(r[i]);
        if (a > rnorm)
            rnorm = a;
    }
    ASSERT_TRUE(rnorm < 1e-10);

    free(p);
    free(ones);
    free(b);
    free(pb);
    free(xp);
    free(x);
    free(r);
    sparse_free(A);
    sparse_free(PA);
}

/* AMD NULL args */
static void test_amd_null(void) {
    idx_t perm[1];
    ASSERT_ERR(sparse_reorder_amd(NULL, perm), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * AMD on SuiteSparse matrices — with comparison to natural & RCM
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: validate AMD on a SuiteSparse matrix, compare fill-in */
static void amd_validate_matrix(const char *path, double res_tol) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, path), SPARSE_OK);
    idx_t n = sparse_rows(A);

    /* Natural ordering fill-in */
    SparseMatrix *LU_nat = sparse_copy(A);
    ASSERT_NOT_NULL(LU_nat);
    ASSERT_ERR(sparse_lu_factor(LU_nat, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    idx_t fill_nat = sparse_nnz(LU_nat);
    sparse_free(LU_nat);

    /* RCM fill-in */
    idx_t *rcm_perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_NOT_NULL(rcm_perm);
    ASSERT_ERR(sparse_reorder_rcm(A, rcm_perm), SPARSE_OK);
    SparseMatrix *PA_rcm = NULL;
    ASSERT_ERR(sparse_permute(A, rcm_perm, rcm_perm, &PA_rcm), SPARSE_OK);
    SparseMatrix *LU_rcm = sparse_copy(PA_rcm);
    ASSERT_NOT_NULL(LU_rcm);
    ASSERT_ERR(sparse_lu_factor(LU_rcm, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    idx_t fill_rcm = sparse_nnz(LU_rcm);
    sparse_free(LU_rcm);
    sparse_free(PA_rcm);
    free(rcm_perm);

    /* AMD */
    idx_t *amd_perm = malloc((size_t)n * sizeof(idx_t));
    ASSERT_NOT_NULL(amd_perm);
    ASSERT_ERR(sparse_reorder_amd(A, amd_perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(amd_perm, n));

    SparseMatrix *PA_amd = NULL;
    ASSERT_ERR(sparse_permute(A, amd_perm, amd_perm, &PA_amd), SPARSE_OK);
    SparseMatrix *LU_amd = sparse_copy(PA_amd);
    ASSERT_NOT_NULL(LU_amd);
    ASSERT_ERR(sparse_lu_factor(LU_amd, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    idx_t fill_amd = sparse_nnz(LU_amd);

    /* Solve with AMD ordering and check residual */
    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *pb = malloc((size_t)n * sizeof(double));
    double *xp = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    double *r = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(pb);
    ASSERT_NOT_NULL(xp);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(r);
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);
    for (idx_t i = 0; i < n; i++)
        pb[i] = b[amd_perm[i]];

    ASSERT_ERR(sparse_lu_solve(LU_amd, pb, xp), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        x[amd_perm[i]] = xp[i];

    sparse_matvec(A, x, r);
    double rnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        r[i] -= b[i];
        double a = fabs(r[i]);
        if (a > rnorm)
            rnorm = a;
    }

    printf("    %s: fill nat=%d rcm=%d amd=%d (%.2fx/%.2fx), res=%.2e\n", path, (int)fill_nat,
           (int)fill_rcm, (int)fill_amd, (double)fill_rcm / (double)fill_nat,
           (double)fill_amd / (double)fill_nat, rnorm);

    ASSERT_TRUE(rnorm < res_tol);

    free(amd_perm);
    free(ones);
    free(b);
    free(pb);
    free(xp);
    free(x);
    free(r);
    sparse_free(A);
    sparse_free(PA_amd);
    sparse_free(LU_amd);
}

static void test_amd_west0067(void) { amd_validate_matrix(SS_DIR "/west0067.mtx", 1e-8); }

static void test_amd_nos4(void) { amd_validate_matrix(SS_DIR "/nos4.mtx", 1e-8); }

static void test_amd_bcsstk04(void) { amd_validate_matrix(SS_DIR "/bcsstk04.mtx", 1e-4); }

static void test_amd_steam1(void) { amd_validate_matrix(SS_DIR "/steam1.mtx", 1e-2); }

/* Stress test: random sparse matrices of increasing size */
static void test_amd_stress(void) {
    /* Use deterministic "random" pattern */
    idx_t sizes[] = {10, 50, 100};
    for (int s = 0; s < 3; s++) {
        idx_t n = sizes[s];
        SparseMatrix *A = sparse_create(n, n);
        /* Diagonal + some off-diagonal entries */
        for (idx_t i = 0; i < n; i++) {
            sparse_insert(A, i, i, (double)(n + 1));
            idx_t j = (i * 7 + 3) % n;
            if (j != i) {
                sparse_insert(A, i, j, 1.0);
                sparse_insert(A, j, i, 1.0);
            }
        }

        idx_t *perm = malloc((size_t)n * sizeof(idx_t));
        ASSERT_NOT_NULL(perm);
        ASSERT_ERR(sparse_reorder_amd(A, perm), SPARSE_OK);
        ASSERT_TRUE(is_valid_perm(perm, n));

        free(perm);
        sparse_free(A);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * sparse_lu_factor_opts integration tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Factor with RCM + partial pivoting → solve → correct result */
static void test_opts_rcm_partial(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    sparse_lu_opts_t opts = {SPARSE_PIVOT_PARTIAL, SPARSE_REORDER_RCM, 1e-12};
    ASSERT_ERR(sparse_lu_factor_opts(LU, &opts), SPARSE_OK);

    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Verify: x should be ~ones (auto-unpermuted) */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-10);

    free(ones);
    free(b);
    free(x);
    sparse_free(A);
    sparse_free(LU);
}

/* Factor with AMD + complete pivoting → solve → correct result */
static void test_opts_amd_complete(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(n + 1));
        if (i > 0) {
            sparse_insert(A, 0, i, 1.0);
            sparse_insert(A, i, 0, 1.0);
        }
    }

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    sparse_lu_opts_t opts = {SPARSE_PIVOT_COMPLETE, SPARSE_REORDER_AMD, 1e-12};
    ASSERT_ERR(sparse_lu_factor_opts(LU, &opts), SPARSE_OK);

    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-10);

    free(ones);
    free(b);
    free(x);
    sparse_free(A);
    sparse_free(LU);
}

/* Factor with NONE → same as sparse_lu_factor */
static void test_opts_none(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 4.0);
    sparse_insert(A, 0, 1, -1.0);
    sparse_insert(A, 1, 0, -1.0);
    sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 1, 2, -1.0);
    sparse_insert(A, 2, 1, -1.0);
    sparse_insert(A, 2, 2, 4.0);

    double b[] = {1.0, 2.0, 3.0};

    /* Solve with opts NONE */
    SparseMatrix *LU1 = sparse_copy(A);
    sparse_lu_opts_t opts = {SPARSE_PIVOT_PARTIAL, SPARSE_REORDER_NONE, 1e-12};
    ASSERT_ERR(sparse_lu_factor_opts(LU1, &opts), SPARSE_OK);
    double x1[3];
    ASSERT_ERR(sparse_lu_solve(LU1, b, x1), SPARSE_OK);

    /* Solve with plain sparse_lu_factor */
    SparseMatrix *LU2 = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU2, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    double x2[3];
    ASSERT_ERR(sparse_lu_solve(LU2, b, x2), SPARSE_OK);

    /* Results should match */
    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x1[i], x2[i], 1e-14);

    sparse_free(A);
    sparse_free(LU1);
    sparse_free(LU2);
}

/* Opts on SuiteSparse matrix */
static void test_opts_suitesparse(void) {
    SparseMatrix *A = NULL;
    ASSERT_ERR(sparse_load_mm(&A, SS_DIR "/west0067.mtx"), SPARSE_OK);
    idx_t n = sparse_rows(A);

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(ones);
    ASSERT_NOT_NULL(b);
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);

    SparseMatrix *LU = sparse_copy(A);
    ASSERT_NOT_NULL(LU);
    sparse_lu_opts_t opts = {SPARSE_PIVOT_PARTIAL, SPARSE_REORDER_AMD, 1e-12};
    ASSERT_ERR(sparse_lu_factor_opts(LU, &opts), SPARSE_OK);

    double *x = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(x);
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);

    /* Verify x ≈ ones */
    double rnorm = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = fabs(x[i] - 1.0);
        if (d > rnorm)
            rnorm = d;
    }
    ASSERT_TRUE(rnorm < 1e-8);

    free(ones);
    free(b);
    free(x);
    sparse_free(A);
    sparse_free(LU);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Edge cases
 * ═══════════════════════════════════════════════════════════════════════ */

/* Rectangular matrix → reordering returns SPARSE_ERR_SHAPE */
static void test_reorder_nonsquare(void) {
    SparseMatrix *A = sparse_create(3, 5);
    idx_t perm[3];
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_ERR_SHAPE);
    ASSERT_ERR(sparse_reorder_amd(A, perm), SPARSE_ERR_SHAPE);
    sparse_free(A);
}

/* 1x1 matrix → reordering is identity */
static void test_reorder_1x1(void) {
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 42.0);

    idx_t perm[1];
    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_EQ(perm[0], 0);

    ASSERT_ERR(sparse_reorder_amd(A, perm), SPARSE_OK);
    ASSERT_EQ(perm[0], 0);

    sparse_free(A);
}

/* Empty matrix (n=0): sparse_create(0,0) returns NULL, reorder returns ERR_NULL */
static void test_reorder_empty(void) {
    SparseMatrix *A = sparse_create(0, 0);
    ASSERT_NULL(A);
    idx_t dummy;
    ASSERT_ERR(sparse_reorder_rcm(A, &dummy), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_reorder_amd(A, &dummy), SPARSE_ERR_NULL);
}

/* Matrix with no off-diagonal entries → permutation is valid */
static void test_reorder_no_offdiag(void) {
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    idx_t *perm = malloc((size_t)n * sizeof(idx_t));

    ASSERT_ERR(sparse_reorder_rcm(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    ASSERT_ERR(sparse_reorder_amd(A, perm), SPARSE_OK);
    ASSERT_TRUE(is_valid_perm(perm, n));

    free(perm);
    sparse_free(A);
}

/* condest on unfactored → BADARG, condest NULL → NULL */
static void test_condest_edge_cases(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);

    double cond;
    /* Unfactored matrix */
    ASSERT_ERR(sparse_lu_condest(A, A, &cond), SPARSE_ERR_BADARG);
    /* NULL args */
    ASSERT_ERR(sparse_lu_condest(NULL, A, &cond), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_lu_condest(A, NULL, &cond), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_lu_condest(A, A, NULL), SPARSE_ERR_NULL);

    sparse_free(A);
}

/* factor_opts with NULL opts → SPARSE_ERR_NULL */
static void test_factor_opts_null(void) {
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 1.0);
    ASSERT_ERR(sparse_lu_factor_opts(A, NULL), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_lu_factor_opts(NULL, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("Reordering & Permutation Tests");

    /* Permute */
    RUN_TEST(test_permute_identity);
    RUN_TEST(test_permute_reverse_diagonal);
    RUN_TEST(test_permute_known);
    RUN_TEST(test_permute_preserves_nnz);
    RUN_TEST(test_permute_null);

    /* Bandwidth */
    RUN_TEST(test_bandwidth_diagonal);
    RUN_TEST(test_bandwidth_tridiag);
    RUN_TEST(test_bandwidth_full);
    RUN_TEST(test_bandwidth_null);

    /* Integration */
    RUN_TEST(test_permute_then_solve);

    /* RCM */
    RUN_TEST(test_rcm_arrow);
    RUN_TEST(test_rcm_tridiag);
    RUN_TEST(test_rcm_diagonal);
    RUN_TEST(test_rcm_disconnected);
    RUN_TEST(test_rcm_solve);

    /* RCM on SuiteSparse matrices */
    RUN_TEST(test_rcm_west0067);
    RUN_TEST(test_rcm_nos4);
    RUN_TEST(test_rcm_bcsstk04);
    RUN_TEST(test_rcm_steam1);

    /* AMD */
    RUN_TEST(test_amd_small);
    RUN_TEST(test_amd_diagonal);
    RUN_TEST(test_amd_solve);
    RUN_TEST(test_amd_null);

    /* AMD on SuiteSparse matrices */
    RUN_TEST(test_amd_west0067);
    RUN_TEST(test_amd_nos4);
    RUN_TEST(test_amd_bcsstk04);
    RUN_TEST(test_amd_steam1);
    RUN_TEST(test_amd_stress);

    /* sparse_lu_factor_opts integration */
    RUN_TEST(test_opts_rcm_partial);
    RUN_TEST(test_opts_amd_complete);
    RUN_TEST(test_opts_none);
    RUN_TEST(test_opts_suitesparse);

    /* Edge cases */
    RUN_TEST(test_reorder_nonsquare);
    RUN_TEST(test_reorder_1x1);
    RUN_TEST(test_reorder_empty);
    RUN_TEST(test_reorder_no_offdiag);
    RUN_TEST(test_condest_edge_cases);
    RUN_TEST(test_factor_opts_null);

    TEST_SUITE_END();
}
