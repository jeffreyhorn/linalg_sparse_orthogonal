/**
 * Sprint 20 cross-feature integration tests.
 *
 * Covers the transparent LDL^T dispatch landed in Days 4-5:
 *   - `sparse_ldlt_opts_t::backend` selector plumbs through
 *     `sparse_ldlt_factor_opts` routing.
 *   - AUTO routes CSC above `SPARSE_CSC_THRESHOLD`, linked-list
 *     below — mirrors the Sprint 18 Cholesky dispatch heuristic.
 *   - Forced CSC on a small matrix and forced LINKED_LIST on a
 *     large matrix each take the requested path regardless of
 *     dimension.
 *   - Indefinite (KKT-style) matrix at n >= threshold factors
 *     end-to-end through the AUTO CSC path and produces a
 *     round-off solve residual — validates Day 3's
 *     `ldlt_csc_from_sparse_with_analysis` plumbing end-to-end
 *     through the public API.
 *   - `used_csc_path` telemetry reports the path correctly on
 *     every branch.
 */

#include "sparse_eigs.h"
#include "sparse_ldlt.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

/* Symmetric SPD tridiagonal, diag 4, off-diag -1 (strictly
 * diagonally dominant).  Small to moderate sizes only — this is
 * the workhorse fixture for cross-threshold routing tests. */
static SparseMatrix *s20_build_spd_tridiag(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

/* Banded SPD with bandwidth bw, strictly diagonally dominant. */
static SparseMatrix *s20_build_spd_banded(idx_t n, idx_t bw) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, (double)(2 * bw + 2));
        for (idx_t d = 1; d <= bw && i + d < n; d++) {
            double off = -1.0 / (double)(d + 1);
            sparse_insert(A, i, i + d, off);
            sparse_insert(A, i + d, i, off);
        }
    }
    return A;
}

/* KKT-style saddle-point indefinite matrix:
 *   [ H    B^T ]
 *   [ B    0   ]
 * H is `n_top`×`n_top` tridiagonal SPD (diag 6, off-diag -1).
 * B = [I_k | 0] where k = n_bot, couples row j+n_top to column j
 * of the top block (j in [0..k)).  Rank-k coupling guarantees the
 * KKT matrix is non-singular for n_top >= n_bot.  Size:
 * n_top + n_bot. */
static SparseMatrix *s20_build_kkt(idx_t n_top, idx_t n_bot) {
    idx_t n = n_top + n_bot;
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n_top; i++) {
        sparse_insert(A, i, i, 6.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    for (idx_t j = 0; j < n_bot; j++) {
        sparse_insert(A, n_top + j, j, 1.0);
        sparse_insert(A, j, n_top + j, 1.0);
    }
    return A;
}

/* Factor + solve a right-hand side `b = A * ones` and return the
 * max-norm residual ||A·x - b||_inf / ||b||_inf.  Returns INFINITY
 * on any intermediate failure. */
static double s20_factor_solve_residual(SparseMatrix *A, const sparse_ldlt_opts_t *opts,
                                        sparse_ldlt_t *ldlt_out) {
    idx_t n = sparse_rows(A);
    if (sparse_ldlt_factor_opts(A, opts, ldlt_out) != SPARSE_OK)
        return INFINITY;

    double *ones = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    double *x = calloc((size_t)n, sizeof(double));
    if (!ones || !b || !x) {
        free(ones);
        free(b);
        free(x);
        return INFINITY;
    }
    for (idx_t i = 0; i < n; i++)
        ones[i] = 1.0;
    sparse_matvec(A, ones, b);
    if (sparse_ldlt_solve(ldlt_out, b, x) != SPARSE_OK) {
        free(ones);
        free(b);
        free(x);
        return INFINITY;
    }
    double *r = malloc((size_t)n * sizeof(double));
    if (!r) {
        free(ones);
        free(b);
        free(x);
        return INFINITY;
    }
    sparse_matvec(A, x, r);
    double nr = 0.0, nb = 0.0;
    for (idx_t i = 0; i < n; i++) {
        double ri = fabs(r[i] - b[i]);
        double bi = fabs(b[i]);
        if (ri > nr)
            nr = ri;
        if (bi > nb)
            nb = bi;
    }
    free(ones);
    free(b);
    free(x);
    free(r);
    return nb > 0.0 ? nr / nb : nr;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Dispatch routing by size (AUTO)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Below-threshold AUTO dispatch: matrix smaller than
 * SPARSE_CSC_THRESHOLD must route to the linked-list kernel.
 * `used_csc_path == 0` and solve residual round-off. */
static void test_s20_auto_below_threshold_routes_linked_list(void) {
    idx_t n = SPARSE_CSC_THRESHOLD / 2;
    ASSERT_TRUE(n >= 2);
    SparseMatrix *A = s20_build_spd_tridiag(n);
    ASSERT_NOT_NULL(A);

    int used_csc = -1;
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_NONE, 0.0, SPARSE_LDLT_BACKEND_AUTO, &used_csc};
    sparse_ldlt_t ldlt;
    double res = s20_factor_solve_residual(A, &opts, &ldlt);
    ASSERT_TRUE(res < 1e-10);
    ASSERT_EQ(used_csc, 0);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* Above-threshold AUTO dispatch on SPD: matrix at SPARSE_CSC_THRESHOLD
 * routes to the CSC supernodal pipeline.  `used_csc_path == 1` and
 * solve residual round-off. */
static void test_s20_auto_above_threshold_spd_routes_csc(void) {
    idx_t n = SPARSE_CSC_THRESHOLD;
    SparseMatrix *A = s20_build_spd_banded(n, 3);
    ASSERT_NOT_NULL(A);

    int used_csc = -1;
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_NONE, 0.0, SPARSE_LDLT_BACKEND_AUTO, &used_csc};
    sparse_ldlt_t ldlt;
    double res = s20_factor_solve_residual(A, &opts, &ldlt);
    ASSERT_TRUE(res < 1e-10);
    ASSERT_EQ(used_csc, 1);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* Above-threshold AUTO dispatch on indefinite KKT: validates the
 * Day 3 `ldlt_csc_from_sparse_with_analysis` enablement end-to-end
 * through the public API.  n = 150 (well above default threshold
 * of 100), KKT structure with 6 rows of coupling into a
 * non-trivial SPD block. */
static void test_s20_auto_above_threshold_indefinite_kkt_routes_csc(void) {
    /* KKT at n = 150: top SPD block 140x140 tridiagonal, bottom
     * zero block 10x10, identity-pattern coupling on first 10
     * columns.  Non-singular by rank-10 Schur complement. */
    SparseMatrix *A = s20_build_kkt(/*n_top=*/140, /*n_bot=*/10);
    ASSERT_NOT_NULL(A);
    ASSERT_EQ(sparse_rows(A), 150);

    int used_csc = -1;
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_NONE, 0.0, SPARSE_LDLT_BACKEND_AUTO, &used_csc};
    sparse_ldlt_t ldlt;
    double res = s20_factor_solve_residual(A, &opts, &ldlt);
    ASSERT_TRUE(res < 1e-10);
    ASSERT_EQ(used_csc, 1);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Forced-path override (CSC / LINKED_LIST explicit)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Forced LINKED_LIST on a large matrix (n >= threshold) takes the
 * linked-list path regardless of the AUTO heuristic.  Residual
 * still round-off. */
static void test_s20_forced_linked_list_on_large_matrix(void) {
    idx_t n = SPARSE_CSC_THRESHOLD + 50;
    SparseMatrix *A = s20_build_spd_banded(n, 3);
    ASSERT_NOT_NULL(A);

    int used_csc = -1;
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_NONE, 0.0, SPARSE_LDLT_BACKEND_LINKED_LIST,
                               &used_csc};
    sparse_ldlt_t ldlt;
    double res = s20_factor_solve_residual(A, &opts, &ldlt);
    ASSERT_TRUE(res < 1e-10);
    ASSERT_EQ(used_csc, 0);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* Forced CSC on a small matrix (n < threshold) takes the CSC path
 * regardless of AUTO's size heuristic. */
static void test_s20_forced_csc_on_small_matrix(void) {
    idx_t n = 10;
    ASSERT_TRUE(n < SPARSE_CSC_THRESHOLD);
    SparseMatrix *A = s20_build_spd_tridiag(n);
    ASSERT_NOT_NULL(A);

    int used_csc = -1;
    sparse_ldlt_opts_t opts = {SPARSE_REORDER_NONE, 0.0, SPARSE_LDLT_BACKEND_CSC, &used_csc};
    sparse_ldlt_t ldlt;
    double res = s20_factor_solve_residual(A, &opts, &ldlt);
    ASSERT_TRUE(res < 1e-10);
    ASSERT_EQ(used_csc, 1);

    sparse_ldlt_free(&ldlt);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 7: sparse_eigs_sym API surface smoke tests
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Day 7 ships the public header (`include/sparse_eigs.h`) and a
 * compile-ready stub (`src/sparse_eigs.c`) returning
 * `SPARSE_ERR_BADARG` on the success path — Days 8-11 replace the
 * stub with the thick-restart Lanczos iteration.  These tests
 * verify the API plumbs through the library build correctly and
 * that the stub enforces the documented preconditions.
 */

/* Validation layer rejects malformed inputs before the stub error
 * path.  Matches the `sparse_eigs_sym` doxygen contract. */
static void test_s20_eigs_rejects_null_args(void) {
    SparseMatrix *A = s20_build_spd_tridiag(4);
    ASSERT_NOT_NULL(A);
    sparse_eigs_t result = {0};
    double vals[3] = {0};
    result.eigenvalues = vals;

    ASSERT_ERR(sparse_eigs_sym(NULL, 3, NULL, &result), SPARSE_ERR_NULL);
    ASSERT_ERR(sparse_eigs_sym(A, 3, NULL, NULL), SPARSE_ERR_NULL);

    /* eigenvalues buffer NULL → SPARSE_ERR_NULL. */
    sparse_eigs_t no_vals = {0};
    ASSERT_ERR(sparse_eigs_sym(A, 3, NULL, &no_vals), SPARSE_ERR_NULL);

    /* compute_vectors set without eigenvectors buffer → SPARSE_ERR_NULL. */
    sparse_eigs_opts_t opts_vecs = {.compute_vectors = 1};
    sparse_eigs_t need_vecs = {.eigenvalues = vals};
    ASSERT_ERR(sparse_eigs_sym(A, 3, &opts_vecs, &need_vecs), SPARSE_ERR_NULL);

    sparse_free(A);
}

/* Shape / range / enum / tol rejections all surface as
 * SPARSE_ERR_SHAPE or SPARSE_ERR_BADARG before the Day 7 stub
 * path fires. */
static void test_s20_eigs_rejects_bad_args(void) {
    SparseMatrix *A = s20_build_spd_tridiag(4);
    double vals[4] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};

    /* k out of range. */
    ASSERT_ERR(sparse_eigs_sym(A, 0, NULL, &result), SPARSE_ERR_BADARG);
    ASSERT_ERR(sparse_eigs_sym(A, 5, NULL, &result), SPARSE_ERR_BADARG);

    /* Invalid which enum. */
    sparse_eigs_opts_t opts_bad_which = {.which = (sparse_eigs_which_t)99};
    ASSERT_ERR(sparse_eigs_sym(A, 2, &opts_bad_which, &result), SPARSE_ERR_BADARG);

    /* Invalid backend enum. */
    sparse_eigs_opts_t opts_bad_backend = {.backend = (sparse_eigs_backend_t)99};
    ASSERT_ERR(sparse_eigs_sym(A, 2, &opts_bad_backend, &result), SPARSE_ERR_BADARG);

    /* Negative tol. */
    sparse_eigs_opts_t opts_neg_tol = {.tol = -1.0};
    ASSERT_ERR(sparse_eigs_sym(A, 2, &opts_neg_tol, &result), SPARSE_ERR_BADARG);

    /* Negative max_iterations. */
    sparse_eigs_opts_t opts_neg_max = {.max_iterations = -1};
    ASSERT_ERR(sparse_eigs_sym(A, 2, &opts_neg_max, &result), SPARSE_ERR_BADARG);

    /* Rectangular → SPARSE_ERR_SHAPE. */
    SparseMatrix *rect = sparse_create(3, 5);
    ASSERT_ERR(sparse_eigs_sym(rect, 2, NULL, &result), SPARSE_ERR_SHAPE);
    sparse_free(rect);

    sparse_free(A);
}

/* Day 7 stub: after validation the function returns
 * SPARSE_ERR_BADARG ("stub in progress" — Days 8-11 replace with
 * Lanczos).  `result->n_requested` is set even on the stub path
 * so callers reading the result struct see a self-describing echo
 * of `k`. */
static void test_s20_eigs_day7_stub_returns_badarg(void) {
    SparseMatrix *A = s20_build_spd_tridiag(6);
    double vals[3] = {0};
    sparse_eigs_t result = {.eigenvalues = vals, .n_requested = 99 /* sentinel */};
    sparse_eigs_opts_t opts = {.which = SPARSE_EIGS_LARGEST, .tol = 1e-10};

    ASSERT_ERR(sparse_eigs_sym(A, 3, &opts, &result), SPARSE_ERR_BADARG);
    /* Validation path zeros these; sentinel 99 overwritten. */
    ASSERT_EQ(result.n_requested, 3);
    ASSERT_EQ(result.n_converged, 0);
    ASSERT_EQ(result.iterations, 0);

    sparse_free(A);
}

/* NULL opts is accepted: library-default opts are used.  Still
 * produces the Day 7 stub error on the success path. */
static void test_s20_eigs_null_opts_uses_defaults(void) {
    SparseMatrix *A = s20_build_spd_tridiag(4);
    double vals[2] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};

    /* Day 7 stub: passes validation, hits the stub return. */
    ASSERT_ERR(sparse_eigs_sym(A, 2, NULL, &result), SPARSE_ERR_BADARG);
    ASSERT_EQ(result.n_requested, 2);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    TEST_SUITE_BEGIN("test_sprint20_integration");

    /* AUTO dispatch routing by size */
    RUN_TEST(test_s20_auto_below_threshold_routes_linked_list);
    RUN_TEST(test_s20_auto_above_threshold_spd_routes_csc);
    RUN_TEST(test_s20_auto_above_threshold_indefinite_kkt_routes_csc);

    /* Forced-path overrides */
    RUN_TEST(test_s20_forced_linked_list_on_large_matrix);
    RUN_TEST(test_s20_forced_csc_on_small_matrix);

    /* Day 7 — sparse_eigs_sym API surface */
    RUN_TEST(test_s20_eigs_rejects_null_args);
    RUN_TEST(test_s20_eigs_rejects_bad_args);
    RUN_TEST(test_s20_eigs_day7_stub_returns_badarg);
    RUN_TEST(test_s20_eigs_null_opts_uses_defaults);

    TEST_SUITE_END();
}
