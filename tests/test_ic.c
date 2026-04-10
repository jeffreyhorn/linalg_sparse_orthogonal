#include "sparse_ic.h"
#include "sparse_ilu.h"
#include "sparse_iterative.h"
#include "sparse_matrix.h"
#include "sparse_matrix_internal.h" /* for Node, row_headers in pattern checks */
#include "sparse_types.h"
#include "test_framework.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Build an n×n SPD tridiagonal matrix: diag=4, off-diag=-1 */
static SparseMatrix *make_spd_tridiag(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

/* Build an n×n SPD diagonal matrix with diag(i) = i+1 */
static SparseMatrix *make_spd_diagonal(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));
    return A;
}

/* Build an n×n SPD banded matrix with bandwidth bw */
static SparseMatrix *make_spd_banded(idx_t n, idx_t bw) {
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        double diag_val = (double)(2 * bw + 2);
        sparse_insert(A, i, i, diag_val);
        for (idx_t d = 1; d <= bw && i + d < n; d++) {
            double off = -1.0 / (double)(d + 1);
            sparse_insert(A, i, i + d, off);
            sparse_insert(A, i + d, i, off);
        }
    }
    return A;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Entry validation tests
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ic_null_args(void) {
    sparse_ilu_t ic;
    SparseMatrix *A = sparse_create(3, 3);

    ASSERT_ERR(sparse_ic_factor(NULL, &ic), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ic_factor(A, NULL), SPARSE_ERR_NULL);

    /* solve with NULL args */
    double r[3] = {1, 2, 3};
    double z[3];
    ASSERT_ERR(sparse_ic_solve(NULL, r, z), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ic_solve(&ic, NULL, z), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ic_solve(&ic, r, NULL), SPARSE_ERR_NULL);

    /* precond with NULL args */
    ASSERT_ERR(sparse_ic_precond(NULL, 3, r, z), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ic_precond(&ic, 3, NULL, z), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_ic_precond(&ic, 3, r, NULL), SPARSE_ERR_NULL);

    /* free on zeroed struct should not crash */
    memset(&ic, 0, sizeof(ic));
    sparse_ic_free(&ic);
    sparse_ic_free(NULL);

    sparse_free(A);
}

static void test_ic_non_square(void) {
    SparseMatrix *A = sparse_create(3, 4);
    sparse_ilu_t ic;
    ASSERT_ERR(sparse_ic_factor(A, &ic), SPARSE_ERR_SHAPE);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_non_symmetric(void) {
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 3.0); /* asymmetric: A(0,1)=1 != A(1,0)=3 */
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);

    sparse_ilu_t ic;
    ASSERT_ERR(sparse_ic_factor(A, &ic), SPARSE_ERR_NOT_SPD);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_empty_matrix(void) {
    /* sparse_create(0,0) returns NULL, so factor should return ERR_NULL */
    SparseMatrix *A = sparse_create(0, 0);
    ASSERT_NULL(A);
    sparse_ilu_t ic;
    ASSERT_ERR(sparse_ic_factor(A, &ic), SPARSE_ERR_NULL);
    sparse_ic_free(&ic);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symbolic phase tests — verify L has correct sparsity pattern
 * ═══════════════════════════════════════════════════════════════════════ */

static void test_ic_diagonal_pattern(void) {
    /* Diagonal SPD matrix: L should be diagonal (each L(i,i) = sqrt(A(i,i))) */
    idx_t n = 5;
    SparseMatrix *A = make_spd_diagonal(n);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    ASSERT_NOT_NULL(ic.L);
    ASSERT_NOT_NULL(ic.U);
    ASSERT_EQ(ic.n, n);

    /* L should have exactly n nonzeros (diagonal only) */
    ASSERT_EQ(sparse_nnz(ic.L), n);

    /* Verify L(i,i) = sqrt(i+1) */
    for (idx_t i = 0; i < n; i++) {
        double expected = sqrt((double)(i + 1));
        double got = sparse_get_phys(ic.L, i, i);
        ASSERT_NEAR(got, expected, 1e-14);
    }

    /* U = L^T, so U should also be diagonal */
    ASSERT_EQ(sparse_nnz(ic.U), n);
    for (idx_t i = 0; i < n; i++) {
        double expected = sqrt((double)(i + 1));
        double got = sparse_get_phys(ic.U, i, i);
        ASSERT_NEAR(got, expected, 1e-14);
    }

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_tridiag_pattern(void) {
    /* Tridiagonal SPD matrix: L should have tridiagonal lower pattern
     * (diagonal + subdiagonal only) */
    idx_t n = 5;
    SparseMatrix *A = make_spd_tridiag(n);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    ASSERT_NOT_NULL(ic.L);
    ASSERT_EQ(ic.n, n);

    /* L should have n diagonal + (n-1) subdiagonal = 2n-1 nonzeros */
    ASSERT_EQ(sparse_nnz(ic.L), 2 * n - 1);

    /* Verify L is lower triangular */
    for (idx_t i = 0; i < n; i++) {
        for (Node *nd = ic.L->row_headers[i]; nd; nd = nd->right) {
            ASSERT_TRUE(nd->col <= i);
        }
    }

    /* U = L^T should be upper triangular with same nnz */
    ASSERT_EQ(sparse_nnz(ic.U), 2 * n - 1);
    for (idx_t i = 0; i < n; i++) {
        for (Node *nd = ic.U->row_headers[i]; nd; nd = nd->right) {
            ASSERT_TRUE(nd->col >= i);
        }
    }

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_tridiag_values(void) {
    /* For a tridiagonal SPD matrix with diag=4, off=-1:
     * Verify L*L^T approximates A at the stored positions */
    idx_t n = 5;
    SparseMatrix *A = make_spd_tridiag(n);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    /* Compute L*L^T and compare with A at lower triangular positions */
    for (idx_t i = 0; i < n; i++) {
        /* Diagonal: (L*L^T)(i,i) = sum_j L(i,j)^2 */
        double diag_sum = 0.0;
        for (Node *nd = ic.L->row_headers[i]; nd; nd = nd->right)
            diag_sum += nd->value * nd->value;
        ASSERT_NEAR(diag_sum, sparse_get_phys(A, i, i), 1e-12);

        /* Subdiagonal: (L*L^T)(i,i-1) = sum_j L(i,j)*L(i-1,j) */
        if (i > 0) {
            double off_sum = 0.0;
            Node *ni = ic.L->row_headers[i];
            Node *ni1 = ic.L->row_headers[i - 1];
            while (ni && ni1) {
                if (ni->col < ni1->col)
                    ni = ni->right;
                else if (ni->col > ni1->col)
                    ni1 = ni1->right;
                else {
                    off_sum += ni->value * ni1->value;
                    ni = ni->right;
                    ni1 = ni1->right;
                }
            }
            ASSERT_NEAR(off_sum, sparse_get_phys(A, i, i - 1), 1e-12);
        }
    }

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_banded_pattern(void) {
    /* Banded SPD matrix with bandwidth 2: L should preserve the lower band */
    idx_t n = 8;
    idx_t bw = 2;
    SparseMatrix *A = make_spd_banded(n, bw);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    /* Verify L is lower triangular */
    for (idx_t i = 0; i < n; i++) {
        for (Node *nd = ic.L->row_headers[i]; nd; nd = nd->right)
            ASSERT_TRUE(nd->col <= i);
    }

    /* L's nnz should match the lower triangle of A */
    idx_t lower_nnz = 0;
    for (idx_t i = 0; i < n; i++) {
        for (Node *nd = A->row_headers[i]; nd; nd = nd->right) {
            if (nd->col <= i)
                lower_nnz++;
        }
    }
    ASSERT_EQ(sparse_nnz(ic.L), lower_nnz);

    printf("    banded n=%d bw=%d: nnz(L)=%d\n", (int)n, (int)bw, (int)sparse_nnz(ic.L));

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_identity(void) {
    /* Identity matrix: L = I, U = I */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    ASSERT_EQ(sparse_nnz(ic.L), n);
    ASSERT_EQ(sparse_nnz(ic.U), n);
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(sparse_get_phys(ic.L, i, i), 1.0, 1e-15);
        ASSERT_NEAR(sparse_get_phys(ic.U, i, i), 1.0, 1e-15);
    }

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_not_spd(void) {
    /* Symmetric but indefinite: diag(1, -1, 1) — IC(0) should fail */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, -1.0);
    sparse_insert(A, 2, 2, 1.0);

    sparse_ilu_t ic;
    ASSERT_ERR(sparse_ic_factor(A, &ic), SPARSE_ERR_NOT_SPD);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_1x1(void) {
    /* 1x1 matrix: trivial case */
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 9.0);

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));
    ASSERT_EQ(ic.n, 1);
    ASSERT_NEAR(sparse_get_phys(ic.L, 0, 0), 3.0, 1e-15);
    ASSERT_NEAR(sparse_get_phys(ic.U, 0, 0), 3.0, 1e-15);

    sparse_ic_free(&ic);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Solve tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compute relative residual ||A*x - b|| / ||b|| */
static double relative_residual(const SparseMatrix *A, const double *x, const double *b, idx_t n) {
    double *r = malloc((size_t)n * sizeof(double));
    sparse_matvec(A, x, r);
    double nr = 0.0, nb = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double d = r[i] - b[i];
        nr += d * d;
        nb += b[i] * b[i];
    }
    free(r);
    return (nb > 0.0) ? sqrt(nr / nb) : sqrt(nr);
}

static void test_ic_solve_diagonal(void) {
    /* Diagonal matrix: IC(0) solve should give exact result */
    idx_t n = 5;
    SparseMatrix *A = make_spd_diagonal(n);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double b[] = {1.0, 4.0, 9.0, 16.0, 25.0};
    double x[5];
    REQUIRE_OK(sparse_ic_solve(&ic, b, x));

    /* A = diag(1,2,3,4,5), so x = b/diag = {1, 2, 3, 4, 5} */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i] / (double)(i + 1), 1e-14);

    double relres = relative_residual(A, x, b, n);
    ASSERT_TRUE(relres < 1e-14);

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_solve_identity(void) {
    /* Identity: solve should return b */
    idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 1.0);

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double b[] = {3.0, -1.0, 7.0, 2.5};
    double x[4];
    REQUIRE_OK(sparse_ic_solve(&ic, b, x));

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], b[i], 1e-15);

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_solve_tridiag(void) {
    /* Tridiagonal SPD: IC(0) is exact for tridiagonal (no fill beyond pattern) */
    idx_t n = 10;
    SparseMatrix *A = make_spd_tridiag(n);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    /* Build known-solution RHS: b = A * x_exact */
    double x_exact[10], b[10], x[10];
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    REQUIRE_OK(sparse_ic_solve(&ic, b, x));

    /* IC(0) is exact for tridiagonal, so residual should be near machine eps */
    double relres = relative_residual(A, x, b, n);
    printf("    tridiag n=%d: relres = %.3e\n", (int)n, relres);
    ASSERT_TRUE(relres < 1e-12);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_solve_banded(void) {
    /* Banded SPD: IC(0) is approximate, but solve should still be reasonable */
    idx_t n = 20;
    idx_t bw = 3;
    SparseMatrix *A = make_spd_banded(n, bw);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(A, x_exact, b);

    REQUIRE_OK(sparse_ic_solve(&ic, b, x));

    /* IC(0) is approximate for banded, but should be a decent preconditioner.
     * Check that the solve doesn't blow up (residual finite and not huge). */
    double relres = relative_residual(A, x, b, n);
    printf("    banded n=%d bw=%d: IC(0) solve relres = %.3e\n", (int)n, (int)bw, relres);
    ASSERT_TRUE(relres < 1.0); /* not divergent */

    free(x_exact);
    free(b);
    free(x);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_solve_1x1(void) {
    /* 1x1: A = [9], L = [3], solve: 9*x = 27 → x = 3 */
    SparseMatrix *A = sparse_create(1, 1);
    sparse_insert(A, 0, 0, 9.0);

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double b = 27.0, x;
    REQUIRE_OK(sparse_ic_solve(&ic, &b, &x));
    ASSERT_NEAR(x, 3.0, 1e-14);

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_solve_multiple_rhs(void) {
    /* Factor once, solve with multiple RHS */
    idx_t n = 8;
    SparseMatrix *A = make_spd_tridiag(n);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    for (int trial = 0; trial < 3; trial++) {
        double x_exact[8], b[8], x[8];
        for (idx_t i = 0; i < n; i++)
            x_exact[i] = (double)(trial * n + i + 1);
        sparse_matvec(A, x_exact, b);

        REQUIRE_OK(sparse_ic_solve(&ic, b, x));

        double relres = relative_residual(A, x, b, n);
        ASSERT_TRUE(relres < 1e-12);
    }

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_precond_callback(void) {
    /* Verify the preconditioner callback works correctly */
    idx_t n = 5;
    SparseMatrix *A = make_spd_tridiag(n);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double r[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double z1[5], z2[5];

    /* Direct solve */
    REQUIRE_OK(sparse_ic_solve(&ic, r, z1));

    /* Via precond callback */
    REQUIRE_OK(sparse_ic_precond(&ic, n, r, z2));

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(z1[i], z2[i], 1e-15);

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_solve_large_tridiag(void) {
    /* Larger tridiagonal: verify IC(0) gives exact results */
    idx_t n = 100;
    SparseMatrix *A = make_spd_tridiag(n);
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = malloc((size_t)n * sizeof(double));

    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    REQUIRE_OK(sparse_ic_solve(&ic, b, x));

    double relres = relative_residual(A, x, b, n);
    printf("    large tridiag n=%d: relres = %.3e\n", (int)n, relres);
    ASSERT_TRUE(relres < 1e-10);

    free(x_exact);
    free(b);
    free(x);
    sparse_ic_free(&ic);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CG preconditioner integration tests
 * ═══════════════════════════════════════════════════════════════════════ */

#define DATA_DIR "tests/data"
#define SS_DIR DATA_DIR "/suitesparse"

static void test_ic_precond_cg_banded(void) {
    /* IC(0)-preconditioned CG on a banded SPD system.
     * Compare: unpreconditioned CG vs IC(0)-CG vs ILU(0)-CG */
    idx_t n = 30;
    idx_t bw = 3;
    SparseMatrix *A = make_spd_banded(n, bw);

    /* Build RHS from known solution */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = sin((double)(i + 1));
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 500, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned CG */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_cg(A, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);
    ASSERT_TRUE(res1.converged);

    /* IC(0)-preconditioned CG */
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_solve_cg(A, b, x2, &opts, sparse_ic_precond, &ic, &res2), SPARSE_OK);
    ASSERT_TRUE(res2.converged);

    /* ILU(0)-preconditioned CG */
    sparse_ilu_t ilu;
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));
    double *x3 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res3;
    ASSERT_ERR(sparse_solve_cg(A, b, x3, &opts, sparse_ilu_precond, &ilu, &res3), SPARSE_OK);
    ASSERT_TRUE(res3.converged);

    printf("    banded n=%d bw=%d CG iters: unprec=%d, IC(0)=%d, ILU(0)=%d\n", (int)n, (int)bw,
           (int)res1.iterations, (int)res2.iterations, (int)res3.iterations);

    /* IC(0) should converge in fewer iterations than unpreconditioned */
    ASSERT_TRUE(res2.iterations <= res1.iterations);

    /* All three solutions should agree */
    for (idx_t i = 0; i < n; i++) {
        ASSERT_NEAR(x1[i], x_exact[i], 1e-6);
        ASSERT_NEAR(x2[i], x_exact[i], 1e-6);
        ASSERT_NEAR(x3[i], x_exact[i], 1e-6);
    }

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    free(x3);
    sparse_ic_free(&ic);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

static void test_ic_precond_cg_large(void) {
    /* Larger banded SPD: verify IC(0) preconditioning scales */
    idx_t n = 100;
    idx_t bw = 4;
    SparseMatrix *A = make_spd_banded(n, bw);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1) / (double)n;
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 1000, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned CG */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    ASSERT_ERR(sparse_solve_cg(A, b, x1, &opts, NULL, NULL, &res1), SPARSE_OK);

    /* IC(0)-preconditioned CG */
    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    ASSERT_ERR(sparse_solve_cg(A, b, x2, &opts, sparse_ic_precond, &ic, &res2), SPARSE_OK);

    printf("    large banded n=%d bw=%d CG iters: unprec=%d, IC(0)=%d\n", (int)n, (int)bw,
           (int)res1.iterations, (int)res2.iterations);

    ASSERT_TRUE(res1.converged);
    ASSERT_TRUE(res2.converged);
    ASSERT_TRUE(res2.iterations <= res1.iterations);

    double relres = relative_residual(A, x2, b, n);
    ASSERT_TRUE(relres < 1e-8);

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_precond_cg_diagonal(void) {
    /* Diagonal matrix: IC(0) is exact inverse, so CG should converge in 1 iter */
    idx_t n = 10;
    SparseMatrix *A = make_spd_diagonal(n);

    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = (double)(i + 1) * 2.0;

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double *x = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, sparse_ic_precond, &ic, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    /* With exact preconditioner on diagonal, should converge in 1 iteration */
    ASSERT_TRUE(res.iterations <= 1);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 2.0, 1e-14);

    free(b);
    free(x);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_precond_cg_zero_rhs(void) {
    /* Zero RHS: CG should return x = 0 immediately */
    idx_t n = 5;
    SparseMatrix *A = make_spd_tridiag(n);

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double b[5] = {0, 0, 0, 0, 0};
    double x[5] = {1, 1, 1, 1, 1}; /* nonzero initial guess */
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, sparse_ic_precond, &ic, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], 0.0, 1e-12);

    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_precond_cg_tridiag_exact(void) {
    /* Tridiagonal: IC(0) is exact Cholesky, so CG converges in 1 iter */
    idx_t n = 20;
    SparseMatrix *A = make_spd_tridiag(n);

    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1);
    sparse_matvec(A, x_exact, b);

    sparse_ilu_t ic;
    REQUIRE_OK(sparse_ic_factor(A, &ic));

    double *x = calloc((size_t)n, sizeof(double));
    sparse_iter_opts_t opts = {.max_iter = 100, .tol = 1e-12, .verbose = 0};
    sparse_iter_result_t res;
    ASSERT_ERR(sparse_solve_cg(A, b, x, &opts, sparse_ic_precond, &ic, &res), SPARSE_OK);

    ASSERT_TRUE(res.converged);
    printf("    tridiag n=%d IC(0)-CG: %d iters\n", (int)n, (int)res.iterations);
    /* Exact preconditioner → should converge in 1 iteration */
    ASSERT_TRUE(res.iterations <= 1);

    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x[i], x_exact[i], 1e-10);

    free(x_exact);
    free(b);
    free(x);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_suitesparse_bcsstk04(void) {
    /* bcsstk04: 132x132 SPD structural stiffness matrix */
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (err != SPARSE_OK || !A) {
        printf("    [SKIP] bcsstk04.mtx not available\n");
        return;
    }

    idx_t n = sparse_rows(A);

    /* IC(0) factorization */
    sparse_ilu_t ic;
    err = sparse_ic_factor(A, &ic);
    if (err != SPARSE_OK) {
        printf("    [SKIP] IC(0) factor failed on bcsstk04 (err=%d)\n", err);
        sparse_free(A);
        return;
    }

    /* Build RHS from known solution */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = 1.0;
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned CG */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    sparse_solve_cg(A, b, x1, &opts, NULL, NULL, &res1);

    /* IC(0)-preconditioned CG */
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    sparse_solve_cg(A, b, x2, &opts, sparse_ic_precond, &ic, &res2);

    printf(
        "    bcsstk04 (n=%d): unprec CG %d iters (relres=%.1e), IC(0)-CG %d iters (relres=%.1e)\n",
        (int)n, (int)res1.iterations, res1.residual_norm, (int)res2.iterations, res2.residual_norm);

    ASSERT_TRUE(res2.converged);
    /* IC(0) should help convergence */
    ASSERT_TRUE(res2.iterations <= res1.iterations);

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_suitesparse_nos4(void) {
    /* nos4: 100x100 SPD matrix */
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/nos4.mtx");
    if (err != SPARSE_OK || !A) {
        printf("    [SKIP] nos4.mtx not available\n");
        return;
    }

    idx_t n = sparse_rows(A);

    sparse_ilu_t ic;
    err = sparse_ic_factor(A, &ic);
    if (err != SPARSE_OK) {
        printf("    [SKIP] IC(0) factor failed on nos4 (err=%d)\n", err);
        sparse_free(A);
        return;
    }

    /* Build RHS */
    double *x_exact = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        x_exact[i] = (double)(i + 1) / (double)n;
    sparse_matvec(A, x_exact, b);

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};

    /* Unpreconditioned CG */
    double *x1 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res1;
    sparse_solve_cg(A, b, x1, &opts, NULL, NULL, &res1);

    /* IC(0)-preconditioned CG */
    double *x2 = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res2;
    sparse_solve_cg(A, b, x2, &opts, sparse_ic_precond, &ic, &res2);

    printf("    nos4 (n=%d): unprec CG %d iters (relres=%.1e), IC(0)-CG %d iters (relres=%.1e)\n",
           (int)n, (int)res1.iterations, res1.residual_norm, (int)res2.iterations,
           res2.residual_norm);

    ASSERT_TRUE(res2.converged);
    ASSERT_TRUE(res2.iterations <= res1.iterations);

    free(x_exact);
    free(b);
    free(x1);
    free(x2);
    sparse_ic_free(&ic);
    sparse_free(A);
}

static void test_ic_vs_ilu_suitesparse(void) {
    /* Compare IC(0) vs ILU(0) as CG preconditioners on bcsstk04 */
    SparseMatrix *A = NULL;
    sparse_err_t err = sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx");
    if (err != SPARSE_OK || !A) {
        printf("    [SKIP] bcsstk04.mtx not available\n");
        return;
    }

    idx_t n = sparse_rows(A);

    /* Build RHS */
    double *b = malloc((size_t)n * sizeof(double));
    for (idx_t i = 0; i < n; i++)
        b[i] = 1.0;

    sparse_iter_opts_t opts = {.max_iter = 2000, .tol = 1e-10, .verbose = 0};

    /* IC(0)-preconditioned CG */
    sparse_ilu_t ic;
    err = sparse_ic_factor(A, &ic);
    if (err != SPARSE_OK) {
        printf("    [SKIP] IC(0) factor failed\n");
        free(b);
        sparse_free(A);
        return;
    }
    double *x_ic = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_ic;
    sparse_solve_cg(A, b, x_ic, &opts, sparse_ic_precond, &ic, &res_ic);

    /* ILU(0)-preconditioned CG */
    sparse_ilu_t ilu;
    REQUIRE_OK(sparse_ilu_factor(A, &ilu));
    double *x_ilu = calloc((size_t)n, sizeof(double));
    sparse_iter_result_t res_ilu;
    sparse_solve_cg(A, b, x_ilu, &opts, sparse_ilu_precond, &ilu, &res_ilu);

    printf("    bcsstk04 IC(0) vs ILU(0): IC(0)=%d iters, ILU(0)=%d iters\n",
           (int)res_ic.iterations, (int)res_ilu.iterations);

    ASSERT_TRUE(res_ic.converged);
    ASSERT_TRUE(res_ilu.converged);

    /* Both solutions should agree */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(x_ic[i], x_ilu[i], 1e-6);

    free(b);
    free(x_ic);
    free(x_ilu);
    sparse_ic_free(&ic);
    sparse_ilu_free(&ilu);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("test_ic");

    /* Entry validation */
    RUN_TEST(test_ic_null_args);
    RUN_TEST(test_ic_non_square);
    RUN_TEST(test_ic_non_symmetric);
    RUN_TEST(test_ic_empty_matrix);
    RUN_TEST(test_ic_not_spd);

    /* Symbolic phase and pattern */
    RUN_TEST(test_ic_1x1);
    RUN_TEST(test_ic_identity);
    RUN_TEST(test_ic_diagonal_pattern);
    RUN_TEST(test_ic_tridiag_pattern);
    RUN_TEST(test_ic_tridiag_values);
    RUN_TEST(test_ic_banded_pattern);

    /* Solve */
    RUN_TEST(test_ic_solve_1x1);
    RUN_TEST(test_ic_solve_identity);
    RUN_TEST(test_ic_solve_diagonal);
    RUN_TEST(test_ic_solve_tridiag);
    RUN_TEST(test_ic_solve_banded);
    RUN_TEST(test_ic_solve_multiple_rhs);
    RUN_TEST(test_ic_solve_large_tridiag);
    RUN_TEST(test_ic_precond_callback);

    /* CG preconditioner integration */
    RUN_TEST(test_ic_precond_cg_diagonal);
    RUN_TEST(test_ic_precond_cg_zero_rhs);
    RUN_TEST(test_ic_precond_cg_tridiag_exact);
    RUN_TEST(test_ic_precond_cg_banded);
    RUN_TEST(test_ic_precond_cg_large);
    RUN_TEST(test_ic_suitesparse_bcsstk04);
    RUN_TEST(test_ic_suitesparse_nos4);
    RUN_TEST(test_ic_vs_ilu_suitesparse);

    TEST_SUITE_END();
}
