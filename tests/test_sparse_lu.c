#include "sparse_matrix.h"
#include "sparse_lu.h"
#include "sparse_types.h"
#include "test_framework.h"
#include <stdlib.h>

/* ─── Helpers ────────────────────────────────────────────────────────── */

/* Compute infinity-norm of a vector */
static double vec_norminf(const double *v, idx_t n)
{
    double mx = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double a = fabs(v[i]);
        if (a > mx) mx = a;
    }
    return mx;
}

/* Check that perm[inv_perm[i]] == i for all i */
static int perms_consistent(const SparseMatrix *m)
{
    idx_t n = sparse_rows(m);
    const idx_t *rp  = sparse_row_perm(m);
    const idx_t *irp = sparse_inv_row_perm(m);
    const idx_t *cp  = sparse_col_perm(m);
    const idx_t *icp = sparse_inv_col_perm(m);
    for (idx_t i = 0; i < n; i++) {
        if (irp[rp[i]] != i) return 0;
        if (rp[irp[i]] != i) return 0;
        if (icp[cp[i]] != i) return 0;
        if (cp[icp[i]] != i) return 0;
    }
    return 1;
}

/*
 * Solve A*x = b with a fresh copy, checking residual.
 * Returns max |A*x - b| using the original matrix.
 * pivot: which pivoting strategy to test.
 */
static double solve_and_residual(SparseMatrix *A_orig, const double *b,
                                 double *x, idx_t n, sparse_pivot_t pivot)
{
    SparseMatrix *A = sparse_copy(A_orig);
    if (!A) return -1.0;

    sparse_err_t err = sparse_lu_factor(A, pivot, 1e-12);
    if (err != SPARSE_OK) { sparse_free(A); return -1.0; }

    err = sparse_lu_solve(A, b, x);
    if (err != SPARSE_OK) { sparse_free(A); return -1.0; }

    /* Compute r = A_orig * x - b */
    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(A_orig, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] -= b[i];
    double res = vec_norminf(r, n);

    free(r);
    sparse_free(A);
    return res;
}

/* Build an n x n identity matrix */
static SparseMatrix *make_identity(idx_t n)
{
    SparseMatrix *m = sparse_create(n, n);
    if (!m) return NULL;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(m, i, i, 1.0);
    return m;
}

/* Build an n x n diagonal matrix with diagonal d[i] = i+1 */
static SparseMatrix *make_diagonal(idx_t n)
{
    SparseMatrix *m = sparse_create(n, n);
    if (!m) return NULL;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(m, i, i, (double)(i + 1));
    return m;
}

/* Build an n x n tridiagonal matrix: -1, 2, -1 (like Poisson 1D) */
static SparseMatrix *make_tridiag(idx_t n)
{
    SparseMatrix *m = sparse_create(n, n);
    if (!m) return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(m, i, i, 2.0);
        if (i > 0)     sparse_insert(m, i, i - 1, -1.0);
        if (i < n - 1) sparse_insert(m, i, i + 1, -1.0);
    }
    return m;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Known solutions
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_solve_1x1(void)
{
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 5.0);
    double b[1] = {10.0}, x[1] = {0};
    double res = solve_and_residual(A, b, x, 1, SPARSE_PIVOT_COMPLETE);
    ASSERT_NEAR(x[0], 2.0, 1e-14);
    ASSERT_NEAR(res, 0.0, 1e-14);
    sparse_free(A);
}

static void test_solve_2x2(void)
{
    /* [2 1] x = [5]  =>  x = [1, 3] */
    /* [1 3]     [10]                  */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);
    sparse_insert(A, 1, 1, 3.0);

    double b[2] = {5.0, 10.0}, x[2];
    double res = solve_and_residual(A, b, x, 2, SPARSE_PIVOT_COMPLETE);
    ASSERT_NEAR(x[0], 1.0, 1e-13);
    ASSERT_NEAR(x[1], 3.0, 1e-13);
    ASSERT_NEAR(res, 0.0, 1e-13);
    sparse_free(A);
}

static void test_solve_3x3(void)
{
    /* The original demo matrix:
     * [1 0 3] x = [1]
     * [0 5 0]     [2]
     * [7 0 9]     [3]
     *
     * det = 1*45 - 3*(-35) = 45+105 = ... wait, let me compute correctly
     * det = 1*(5*9 - 0*0) - 0 + 3*(0 - 5*7) = 45 - 105 = -60
     *    (expanding along row 0, but col 1 is zero)
     *    actually: 1*(5*9) - 0 + 3*(0 - 5*7) = 45 - 105 = ... hmm
     *    No: det = 1*(5*9 - 0*0) - 0*(0*9 - 0*7) + 3*(0*0 - 5*7)
     *           = 45 + 3*(-35) = 45 - 105 = -60
     * x = A^{-1} b. Just check residual.
     */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0);
    sparse_insert(A, 2, 2, 9.0);

    double b[3] = {1.0, 2.0, 3.0}, x[3];
    double res = solve_and_residual(A, b, x, 3, SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-13);
    sparse_free(A);
}

static void test_solve_4x4(void)
{
    /* Dense 4x4:
     * [2  1  0  0]     [1]
     * [1  3  1  0] x = [2]
     * [0  1  4  1]     [3]
     * [0  0  1  5]     [4]
     */
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 2.0); sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0); sparse_insert(A, 1, 1, 3.0);
    sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0); sparse_insert(A, 2, 2, 4.0);
    sparse_insert(A, 2, 3, 1.0);
    sparse_insert(A, 3, 2, 1.0); sparse_insert(A, 3, 3, 5.0);

    double b[4] = {1.0, 2.0, 3.0, 4.0}, x[4];
    double res = solve_and_residual(A, b, x, 4, SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-12);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Identity matrix
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_identity(void)
{
    SparseMatrix *I = make_identity(5);
    SparseMatrix *LU = sparse_copy(I);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);

    /* After factoring identity: L=I (implicit), U=I */
    for (idx_t i = 0; i < 5; i++)
        for (idx_t j = 0; j < 5; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(sparse_get(LU, i, j), expected, 1e-14);
        }

    /* Solve I*x = b => x = b */
    double b[5] = {10, 20, 30, 40, 50}, x[5];
    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);
    for (int i = 0; i < 5; i++)
        ASSERT_NEAR(x[i], b[i], 1e-14);

    sparse_free(LU);
    sparse_free(I);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Diagonal matrix
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_diagonal(void)
{
    idx_t n = 5;
    SparseMatrix *D = make_diagonal(n);
    double b[5], x[5];
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1) * 10.0;  /* b[i] = 10, 20, 30, 40, 50 */

    double res = solve_and_residual(D, b, x, n, SPARSE_PIVOT_COMPLETE);
    ASSERT_NEAR(res, 0.0, 1e-13);

    /* x[i] = b[i] / d[i] = 10*(i+1) / (i+1) = 10 for all i */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 10.0, 1e-13);

    sparse_free(D);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Triangular matrices
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_upper_triangular(void)
{
    /* Upper triangular: LU should give L=I, U=A (up to pivoting) */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0); sparse_insert(A, 0, 1, 1.0); sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 4.0); sparse_insert(A, 1, 2, 5.0);
    sparse_insert(A, 2, 2, 6.0);

    double b[3] = {1.0, 2.0, 3.0}, x[3];
    double res = solve_and_residual(A, b, x, 3, SPARSE_PIVOT_COMPLETE);
    ASSERT_NEAR(res, 0.0, 1e-13);
    sparse_free(A);
}

static void test_lu_lower_triangular(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 3.0);
    sparse_insert(A, 1, 0, 2.0); sparse_insert(A, 1, 1, 4.0);
    sparse_insert(A, 2, 0, 1.0); sparse_insert(A, 2, 1, 5.0); sparse_insert(A, 2, 2, 6.0);

    double b[3] = {3.0, 10.0, 28.0}, x[3];
    double res = solve_and_residual(A, b, x, 3, SPARSE_PIVOT_COMPLETE);
    ASSERT_NEAR(res, 0.0, 1e-13);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Tridiagonal (Poisson 1D)
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_tridiag(void)
{
    idx_t n = 20;
    SparseMatrix *A = make_tridiag(n);
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    double res = solve_and_residual(A, b, x, n, SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_NEAR(res, 0.0, 1e-11);

    free(b); free(x);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Permutation consistency
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_perm_consistency_after_lu(void)
{
    SparseMatrix *A = sparse_create(4, 4);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0); sparse_insert(A, 1, 3, 2.0);
    sparse_insert(A, 2, 0, 7.0); sparse_insert(A, 2, 2, 9.0);
    sparse_insert(A, 3, 1, 4.0); sparse_insert(A, 3, 3, 6.0);

    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);
    ASSERT_TRUE(perms_consistent(A));
    sparse_free(A);
}

static void test_perm_consistency_partial_pivot(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 2, 3.0);
    sparse_insert(A, 1, 1, 5.0);
    sparse_insert(A, 2, 0, 7.0); sparse_insert(A, 2, 2, 9.0);

    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_PARTIAL, 1e-12), SPARSE_OK);
    ASSERT_TRUE(perms_consistent(A));

    /* Partial pivoting: col perm should be identity */
    const idx_t *cp = sparse_col_perm(A);
    for (idx_t i = 0; i < 3; i++)
        ASSERT_EQ(cp[i], i);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Residual checks on larger matrices
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_residual_10x10(void)
{
    idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);

    /* Dense-ish 10x10 with a known pattern */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            double val = 1.0 / (double)(i + j + 1);  /* Hilbert-like */
            if (fabs(val) > 0.05)
                sparse_insert(A, i, j, val);
        }
        /* Ensure diagonal dominance */
        sparse_insert(A, i, i, (double)(n + 1));
    }

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1);

    double res = solve_and_residual(A, b, x, n, SPARSE_PIVOT_COMPLETE);
    ASSERT_TRUE(res >= 0.0);
    ASSERT_TRUE(res < 1e-10);  /* relaxed for ill-conditioning */

    free(b); free(x);
    sparse_free(A);
}

static void test_both_pivot_strategies_agree(void)
{
    /* Both strategies should give close-enough solutions */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0); sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0); sparse_insert(A, 1, 1, 3.0); sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0); sparse_insert(A, 2, 2, 4.0);

    double b[3] = {3.0, 5.0, 5.0};
    double x_complete[3], x_partial[3];

    solve_and_residual(A, b, x_complete, 3, SPARSE_PIVOT_COMPLETE);
    solve_and_residual(A, b, x_partial, 3, SPARSE_PIVOT_PARTIAL);

    for (int i = 0; i < 3; i++)
        ASSERT_NEAR(x_complete[i], x_partial[i], 1e-12);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Singular detection
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_singular_zero_matrix(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    /* All zeros — nnz=0 */
    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12),
               SPARSE_ERR_SINGULAR);
    sparse_free(A);
}

static void test_singular_rank_deficient(void)
{
    /* Row 1 = 2 * Row 0 */
    SparseMatrix *A = sparse_create(2, 2);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0); sparse_insert(A, 1, 1, 4.0);

    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12),
               SPARSE_ERR_SINGULAR);
    sparse_free(A);
}

static void test_singular_zero_row(void)
{
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0); sparse_insert(A, 0, 1, 2.0); sparse_insert(A, 0, 2, 3.0);
    /* row 1 all zeros */
    sparse_insert(A, 2, 0, 4.0); sparse_insert(A, 2, 1, 5.0); sparse_insert(A, 2, 2, 6.0);

    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12),
               SPARSE_ERR_SINGULAR);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Error path tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_lu_null_matrix(void)
{
    ASSERT_ERR(sparse_lu_factor(NULL, SPARSE_PIVOT_COMPLETE, 1e-12),
               SPARSE_ERR_NULL);
}

static void test_lu_nonsquare(void)
{
    SparseMatrix *A = sparse_create(3, 5);
    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12),
               SPARSE_ERR_SHAPE);
    sparse_free(A);
}

static void test_solve_null_args(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    double b[2] = {1, 2}, x[2];

    ASSERT_ERR(sparse_lu_solve(NULL, b, x), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_lu_solve(A, NULL, x), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_lu_solve(A, b, NULL), SPARSE_ERR_NULL);
    sparse_free(A);
}

static void test_forward_sub_null(void)
{
    ASSERT_ERR(sparse_forward_sub(NULL, NULL, NULL), SPARSE_ERR_NULL);
}

static void test_backward_sub_null(void)
{
    ASSERT_ERR(sparse_backward_sub(NULL, NULL, NULL), SPARSE_ERR_NULL);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Iterative refinement
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_iterative_refinement(void)
{
    idx_t n = 10;
    SparseMatrix *A = make_tridiag(n);
    SparseMatrix *LU = sparse_copy(A);
    ASSERT_ERR(sparse_lu_factor(LU, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);

    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    ASSERT_ERR(sparse_lu_solve(LU, b, x), SPARSE_OK);
    ASSERT_ERR(sparse_lu_refine(A, LU, b, x, 5, 1e-15), SPARSE_OK);

    /* Check residual after refinement */
    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(A, x, r);
    for (idx_t i = 0; i < n; i++)
        r[i] -= b[i];
    double res = vec_norminf(r, n);
    ASSERT_TRUE(res < 1e-14);

    free(r); free(b); free(x);
    sparse_free(LU);
    sparse_free(A);
}

static void test_refine_null_args(void)
{
    SparseMatrix *A = sparse_create(2, 2);
    double b[2], x[2];
    ASSERT_ERR(sparse_lu_refine(NULL, A, b, x, 1, 1e-10), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_lu_refine(A, NULL, b, x, 1, 1e-10), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_lu_refine(A, A, NULL, x, 1, 1e-10), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_lu_refine(A, A, b, NULL, 1, 1e-10), SPARSE_ERR_NULL);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Drop tolerance
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_drop_tolerance_reduces_fillin(void)
{
    /* Use a matrix where factorization creates fill-in, and verify
     * nnz after factorization doesn't explode to n*n */
    idx_t n = 20;
    SparseMatrix *A = make_tridiag(n);
    idx_t nnz_before = sparse_nnz(A);

    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_COMPLETE, 1e-12), SPARSE_OK);

    idx_t nnz_after = sparse_nnz(A);
    /* For tridiagonal, fill-in should be modest */
    ASSERT_TRUE(nnz_after < n * n);
    ASSERT_TRUE(nnz_after >= nnz_before);  /* can't lose entries */
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Relative drop tolerance
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_relative_tol_scaled_identity(void)
{
    /*
     * Scale identity by 1e-16. With absolute DROP_TOL=1e-14, the diagonal
     * entries (1e-16) would be smaller than DROP_TOL, triggering a false
     * singular detection. With relative tolerance (DROP_TOL * ||A||_inf),
     * the threshold becomes 1e-14 * 1e-16 = 1e-30, so 1e-16 passes.
     */
    idx_t n = 5;
    double scale = 1e-16;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, scale);

    SparseMatrix *A_copy = sparse_copy(A);

    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_PARTIAL, 1e-30), SPARSE_OK);

    /* Solve Ax = b where b = scale * [1,1,...,1], expect x = [1,1,...,1] */
    double b[5], x[5];
    for (idx_t i = 0; i < n; i++) b[i] = scale;

    ASSERT_ERR(sparse_lu_solve(A, b, x), SPARSE_OK);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 1.0, 1e-10);

    sparse_free(A);
    sparse_free(A_copy);
}

static void test_relative_tol_singular_still_detected(void)
{
    /* A genuinely singular matrix should still be caught */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 0, 1, 2.0);
    sparse_insert(A, 1, 0, 2.0);
    sparse_insert(A, 1, 1, 4.0);  /* row 1 = 2 * row 0 */
    sparse_insert(A, 2, 2, 1.0);

    ASSERT_ERR(sparse_lu_factor(A, SPARSE_PIVOT_PARTIAL, 1e-12),
               SPARSE_ERR_SINGULAR);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test runner
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    TEST_SUITE_BEGIN("Sparse LU Factorization Tests");

    /* Known solutions */
    RUN_TEST(test_solve_1x1);
    RUN_TEST(test_solve_2x2);
    RUN_TEST(test_solve_3x3);
    RUN_TEST(test_solve_4x4);

    /* Special matrices */
    RUN_TEST(test_lu_identity);
    RUN_TEST(test_lu_diagonal);
    RUN_TEST(test_lu_upper_triangular);
    RUN_TEST(test_lu_lower_triangular);
    RUN_TEST(test_lu_tridiag);

    /* Permutation consistency */
    RUN_TEST(test_perm_consistency_after_lu);
    RUN_TEST(test_perm_consistency_partial_pivot);

    /* Residual checks */
    RUN_TEST(test_residual_10x10);
    RUN_TEST(test_both_pivot_strategies_agree);

    /* Singular detection */
    RUN_TEST(test_singular_zero_matrix);
    RUN_TEST(test_singular_rank_deficient);
    RUN_TEST(test_singular_zero_row);

    /* Error paths */
    RUN_TEST(test_lu_null_matrix);
    RUN_TEST(test_lu_nonsquare);
    RUN_TEST(test_solve_null_args);
    RUN_TEST(test_forward_sub_null);
    RUN_TEST(test_backward_sub_null);

    /* Iterative refinement */
    RUN_TEST(test_iterative_refinement);
    RUN_TEST(test_refine_null_args);

    /* Drop tolerance */
    RUN_TEST(test_drop_tolerance_reduces_fillin);

    /* Relative tolerance */
    RUN_TEST(test_relative_tol_scaled_identity);
    RUN_TEST(test_relative_tol_singular_still_detected);

    TEST_SUITE_END();
}
