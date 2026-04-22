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

#include "sparse_dense.h"
#include "sparse_eigs.h"
#include "sparse_ldlt.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"

/* Internal — exercises Lanczos helpers directly ahead of Day 11's
 * `sparse_eigs_sym` implementation wiring. */
#include "sparse_eigs_internal.h"

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

/* Day 10 replaces the Day 7 stub with the full Lanczos body.
 * On a well-formed input, `sparse_eigs_sym` now returns
 * SPARSE_OK and populates result->eigenvalues / n_converged /
 * iterations / residual_norm.  The result struct's self-
 * describing echo of `k` (n_requested) is set regardless. */
static void test_s20_eigs_well_formed_call_succeeds(void) {
    SparseMatrix *A = s20_build_spd_tridiag(6);
    double vals[3] = {0};
    sparse_eigs_t result = {.eigenvalues = vals, .n_requested = 99 /* sentinel */};
    sparse_eigs_opts_t opts = {.which = SPARSE_EIGS_LARGEST, .tol = 1e-10};

    REQUIRE_OK(sparse_eigs_sym(A, 3, &opts, &result));
    ASSERT_EQ(result.n_requested, 3);
    ASSERT_EQ(result.n_converged, 3);
    ASSERT_TRUE(result.iterations > 0);

    sparse_free(A);
}

/* NULL opts is accepted: library-default opts are used.  Day 10
 * returns SPARSE_OK with all requested Ritz values populated. */
static void test_s20_eigs_null_opts_uses_defaults(void) {
    SparseMatrix *A = s20_build_spd_tridiag(4);
    double vals[2] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};

    REQUIRE_OK(sparse_eigs_sym(A, 2, NULL, &result));
    ASSERT_EQ(result.n_requested, 2);
    ASSERT_EQ(result.n_converged, 2);

    sparse_free(A);
}

/* Eigenvector output requested but not supported yet.  Day 10
 * scopes this out; Day 11 adds Y computation. */
static void test_s20_eigs_compute_vectors_rejected_day10(void) {
    SparseMatrix *A = s20_build_spd_tridiag(4);
    double vals[2] = {0};
    double vecs[4 * 2] = {0};
    sparse_eigs_t result = {.eigenvalues = vals, .eigenvectors = vecs};
    sparse_eigs_opts_t opts = {.compute_vectors = 1};
    ASSERT_ERR(sparse_eigs_sym(A, 2, &opts, &result), SPARSE_ERR_BADARG);
    sparse_free(A);
}

/* NEAREST_SIGMA (shift-invert) deferred to Day 12; Day 10 rejects. */
static void test_s20_eigs_nearest_sigma_rejected_day10(void) {
    SparseMatrix *A = s20_build_spd_tridiag(4);
    double vals[2] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {.which = SPARSE_EIGS_NEAREST_SIGMA, .sigma = 1.0};
    ASSERT_ERR(sparse_eigs_sym(A, 2, &opts, &result), SPARSE_ERR_BADARG);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 8: Lanczos 3-term recurrence
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Day 8 lands `lanczos_iterate` — the basic 3-term Lanczos
 * recurrence without reorthogonalization.  These tests verify the
 * recurrence builds a valid Lanczos basis + tridiagonal T by
 * cross-checking T's spectrum against A's on fixtures where the
 * answer is known exactly.
 */

/* Diagonal matrix A = diag(1, 2, ..., n) with a "random-looking"
 * starting vector.  Lanczos for m = n iterations builds a full
 * Krylov basis of dimension n; T = V^T·A·V has exactly the same
 * spectrum as A.  Use `tridiag_qr_eigenvalues` to extract T's
 * eigenvalues and assert they match {1, 2, ..., n} to 1e-10. */
static void test_s20_day8_lanczos_diagonal_spectrum(void) {
    const idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    /* Deterministic non-sparse starting vector — avoids alignment
     * with any single eigenvector that would truncate the Krylov
     * basis early. */
    double v0[6] = {1.0, 0.5, -0.3, 0.7, 0.2, -0.9};
    double V[6 * 6];
    double alpha[6];
    double beta[6];
    idx_t m_actual = 0;

    REQUIRE_OK(lanczos_iterate(A, v0, n, /*reorthogonalize=*/0, V, alpha, beta, &m_actual));
    ASSERT_EQ(m_actual, n);

    /* Extract T's spectrum via the existing symmetric-tridiagonal
     * QR solver.  It's destructive — copy alpha/beta into local
     * scratch first.  tridiag_qr_eigenvalues takes diag (length n)
     * and subdiag (length n-1); Lanczos's beta[0..n-2] is the
     * tridiagonal super/subdiagonal, and beta[n-1] is the final
     * residual norm (not part of T). */
    double tdiag[6];
    double tsub[5];
    for (idx_t i = 0; i < n; i++)
        tdiag[i] = alpha[i];
    for (idx_t i = 0; i < n - 1; i++)
        tsub[i] = beta[i];

    REQUIRE_OK(tridiag_qr_eigenvalues(tdiag, tsub, n, 0));

    /* tridiag_qr_eigenvalues returns ascending eigenvalues in tdiag. */
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(tdiag[i], (double)(i + 1), 1e-10);

    sparse_free(A);
}

/* Canonical tridiagonal A (diag = 2, sub/super-diag = -1).
 * With v0 = e_0, Lanczos on A reproduces A's tridiagonal structure
 * exactly: every alpha_k = A[k, k] = 2, every beta_k = |A[k, k+1]|
 * = 1 (betas are non-negative by construction: beta = ||w||).
 * The iteration terminates with beta_{n-1} ≈ 0 because the Krylov
 * basis span(e_0, A·e_0, ..., A^{n-1}·e_0) has dimension exactly
 * n on this fixture. */
static void test_s20_day8_lanczos_tridiagonal_identity(void) {
    const idx_t n = 5;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    double v0[5] = {1.0, 0.0, 0.0, 0.0, 0.0};
    double V[5 * 5];
    double alpha[5];
    double beta[5];
    idx_t m_actual = 0;

    REQUIRE_OK(lanczos_iterate(A, v0, n, /*reorthogonalize=*/0, V, alpha, beta, &m_actual));
    ASSERT_EQ(m_actual, n);

    for (idx_t k = 0; k < n; k++)
        ASSERT_NEAR(alpha[k], 2.0, 1e-12);
    for (idx_t k = 0; k < n - 1; k++)
        ASSERT_NEAR(beta[k], 1.0, 1e-12);
    /* The final step computes w -> 0 (invariant subspace reached);
     * early-exit records the last beta as ≈ 0, then returns. */
    ASSERT_TRUE(fabs(beta[n - 1]) < 1e-12);

    sparse_free(A);
}

/* Argument-validation paths.  Covers the SPARSE_ERR_NULL /
 * SHAPE / BADARG branches before the main recurrence body. */
static void test_s20_day8_lanczos_rejects_bad_args(void) {
    SparseMatrix *A = s20_build_spd_tridiag(4);
    double v0[4] = {1, 0, 0, 0};
    double V[4 * 4];
    double alpha[4];
    double beta[4];
    idx_t m_actual = 99;

    ASSERT_ERR(lanczos_iterate(NULL, v0, 4, 0, V, alpha, beta, &m_actual), SPARSE_ERR_NULL);
    ASSERT_ERR(lanczos_iterate(A, NULL, 4, 0, V, alpha, beta, &m_actual), SPARSE_ERR_NULL);
    ASSERT_ERR(lanczos_iterate(A, v0, 4, 0, NULL, alpha, beta, &m_actual), SPARSE_ERR_NULL);
    ASSERT_ERR(lanczos_iterate(A, v0, 4, 0, V, NULL, beta, &m_actual), SPARSE_ERR_NULL);
    ASSERT_ERR(lanczos_iterate(A, v0, 4, 0, V, alpha, NULL, &m_actual), SPARSE_ERR_NULL);
    ASSERT_ERR(lanczos_iterate(A, v0, 4, 0, V, alpha, beta, NULL), SPARSE_ERR_NULL);

    /* m_max out of range. */
    ASSERT_ERR(lanczos_iterate(A, v0, 0, 0, V, alpha, beta, &m_actual), SPARSE_ERR_BADARG);
    ASSERT_ERR(lanczos_iterate(A, v0, 5, 0, V, alpha, beta, &m_actual), SPARSE_ERR_BADARG);

    /* Zero starting vector. */
    double v_zero[4] = {0, 0, 0, 0};
    ASSERT_ERR(lanczos_iterate(A, v_zero, 4, 0, V, alpha, beta, &m_actual), SPARSE_ERR_BADARG);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 9: Lanczos full reorthogonalization
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Day 9 adds the `reorthogonalize` gate to `lanczos_iterate`.
 * These tests verify that with reorth enabled the Lanczos basis
 * V maintains ‖V^T·V − I‖_max ≤ 1e-10 on a wide-spectrum SPD
 * fixture where the basic recurrence is known to lose
 * orthogonality (Paige/Parlett "ghost eigenvalues" regime).
 */

/* Diagonally-dominant dense SPD with exponentially-spaced
 * diagonal entries.  A = diag(s_0, s_1, ..., s_{n-1}) +
 * 0.01 * (1/(1 + |i-j|)) on off-diagonals (symmetric); the spread
 * s_i = 10^(i * spread_decades / (n-1)) delivers condition
 * numbers in the 1e6+ range that exercise the Lanczos
 * orthogonality-loss behaviour. */
static SparseMatrix *s20_day9_build_wide_spectrum(idx_t n, double spread_decades) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        double s = pow(10.0, spread_decades * (double)i / (double)(n - 1));
        sparse_insert(A, i, i, s);
    }
    /* Small symmetric perturbation so A isn't diagonal (prevents
     * alignment with e_i Krylov bases that would terminate early). */
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = i + 1; j < n; j++) {
            double v = 0.01 / (double)(j - i + 1);
            sparse_insert(A, i, j, v);
            sparse_insert(A, j, i, v);
        }
    }
    return A;
}

/* Compute max |(V^T V)[i, j] - I[i, j]| over the first m columns
 * of V (which is stored column-major as an n x m_cap matrix).
 * This is the Day 9 orthogonality-loss metric. */
static double s20_day9_ortho_drift(const double *V, idx_t n, idx_t m) {
    double drift = 0.0;
    for (idx_t i = 0; i < m; i++) {
        const double *v_i = V + i * n;
        for (idx_t j = 0; j <= i; j++) {
            const double *v_j = V + j * n;
            double dot = 0.0;
            for (idx_t k = 0; k < n; k++)
                dot += v_i[k] * v_j[k];
            double target = (i == j) ? 1.0 : 0.0;
            double err = fabs(dot - target);
            if (err > drift)
                drift = err;
        }
    }
    return drift;
}

/* Deterministic non-sparse starting vector of length n; avoids
 * alignment with any single eigenvector and exercises every
 * eigendirection. */
static void s20_day9_fill_v0(double *v0, idx_t n) {
    for (idx_t i = 0; i < n; i++) {
        /* Pseudo-random mixing so v0 has components in every
         * eigendirection of a diagonal fixture.  Deterministic
         * so reruns produce identical numbers. */
        double x = (double)(i + 1) * 0.618033988749895;
        v0[i] = 0.3 + (x - floor(x));
    }
}

/* With full reorthogonalization enabled, V^T V stays at ≈ I on a
 * wide-spectrum n=100 SPD (condition ≈ 10^6) through 40 Lanczos
 * iterations. */
static void test_s20_day9_reorth_maintains_orthogonality(void) {
    idx_t n = 100;
    idx_t m = 40;
    SparseMatrix *A = s20_day9_build_wide_spectrum(n, /*spread_decades=*/6.0);
    ASSERT_NOT_NULL(A);

    double *v0 = malloc((size_t)n * sizeof(double));
    double *V = malloc((size_t)n * (size_t)m * sizeof(double));
    double *alpha = malloc((size_t)m * sizeof(double));
    double *beta = malloc((size_t)m * sizeof(double));
    ASSERT_NOT_NULL(v0);
    ASSERT_NOT_NULL(V);
    ASSERT_NOT_NULL(alpha);
    ASSERT_NOT_NULL(beta);
    s20_day9_fill_v0(v0, n);

    idx_t m_actual = 0;
    REQUIRE_OK(lanczos_iterate(A, v0, m, /*reorthogonalize=*/1, V, alpha, beta, &m_actual));
    ASSERT_EQ(m_actual, m);

    double drift = s20_day9_ortho_drift(V, n, m_actual);
    ASSERT_TRUE(drift < 1e-10);

    free(v0);
    free(V);
    free(alpha);
    free(beta);
    sparse_free(A);
}

/* Complementary sanitizer exercise: running the same fixture with
 * reorth disabled must still succeed (return SPARSE_OK).  The
 * orthogonality drift may be large (ghost Ritz values are
 * expected — Paige/Parlett known behavior) but the iteration
 * must not crash, overflow, or leak memory.  This is how Day 9
 * exercises the reorth-off path under ASan/UBSan. */
static void test_s20_day9_no_reorth_completes_cleanly(void) {
    idx_t n = 100;
    idx_t m = 40;
    SparseMatrix *A = s20_day9_build_wide_spectrum(n, 6.0);
    ASSERT_NOT_NULL(A);

    double *v0 = malloc((size_t)n * sizeof(double));
    double *V = malloc((size_t)n * (size_t)m * sizeof(double));
    double *alpha = malloc((size_t)m * sizeof(double));
    double *beta = malloc((size_t)m * sizeof(double));
    ASSERT_NOT_NULL(v0);
    ASSERT_NOT_NULL(V);
    ASSERT_NOT_NULL(alpha);
    ASSERT_NOT_NULL(beta);
    s20_day9_fill_v0(v0, n);

    idx_t m_actual = 0;
    REQUIRE_OK(lanczos_iterate(A, v0, m, /*reorthogonalize=*/0, V, alpha, beta, &m_actual));
    ASSERT_EQ(m_actual, m);
    /* Do not assert drift — the point of this test is that the
     * non-reorth path runs to completion cleanly, not that it
     * retains orthogonality.  Day 11 will handle ghost Ritz
     * values via thick-restart + convergence filtering. */

    free(v0);
    free(V);
    free(alpha);
    free(beta);
    sparse_free(A);
}

/* On a small well-conditioned fixture, reorth-on and reorth-off
 * produce *numerically identical* alpha values (up to 1e-12)
 * because orthogonality is already preserved by the 3-term
 * recurrence alone.  Validates the reorth path doesn't
 * accidentally perturb the T spectrum on easy inputs. */
static void test_s20_day9_reorth_agrees_with_basic_on_small_matrix(void) {
    idx_t n = 8;
    SparseMatrix *A = sparse_create(n, n);
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    double v0[8];
    s20_day9_fill_v0(v0, n);

    double V_basic[8 * 8], alpha_basic[8], beta_basic[8];
    double V_reorth[8 * 8], alpha_reorth[8], beta_reorth[8];
    idx_t m_b = 0, m_r = 0;

    REQUIRE_OK(lanczos_iterate(A, v0, n, 0, V_basic, alpha_basic, beta_basic, &m_b));
    REQUIRE_OK(lanczos_iterate(A, v0, n, 1, V_reorth, alpha_reorth, beta_reorth, &m_r));

    ASSERT_EQ(m_b, m_r);
    for (idx_t k = 0; k < m_b; k++) {
        ASSERT_NEAR(alpha_basic[k], alpha_reorth[k], 1e-12);
        ASSERT_NEAR(beta_basic[k], beta_reorth[k], 1e-12);
    }

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 10: thick-restart Lanczos — end-to-end `sparse_eigs_sym`
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Day 10 replaces the Day 7 stub body of `sparse_eigs_sym` with
 * the outer-loop driver documented in the thick-restart Lanczos
 * design block in `src/sparse_eigs.c`.  These tests verify the
 * public API converges to k extreme eigenvalues within the
 * iteration budget, matching an analytically-known reference on
 * canonical SPD fixtures.
 */

/* n=200 SPD with well-separated top/bottom eigenvalues: a
 * diagonal with most entries in [1, 100] + 5 "spikes" at both
 * ends of the spectrum.  Lanczos converges to the extreme pairs
 * quickly because the spectral gaps are large, matching the
 * "moderate-size SPD" the Day 10 plan intends. */
static SparseMatrix *s20_day10_build_gapped_spd(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    /* Background spectrum: 1 + small linear ramp in [0, 1) so all
     * interior eigenvalues sit in [1, 2]. */
    for (idx_t i = 0; i < n - 10; i++) {
        double s = 1.0 + 1.0 * (double)i / (double)(n - 11);
        sparse_insert(A, i, i, s);
    }
    /* Top 5 spikes: 100, 200, 400, 800, 1600 (gap ratio 2×). */
    sparse_insert(A, n - 1, n - 1, 1600.0);
    sparse_insert(A, n - 2, n - 2, 800.0);
    sparse_insert(A, n - 3, n - 3, 400.0);
    sparse_insert(A, n - 4, n - 4, 200.0);
    sparse_insert(A, n - 5, n - 5, 100.0);
    /* Bottom 5 spikes: 1e-4, 2e-4, 4e-4, 8e-4, 16e-4 (well
     * below the [1, 2] background band). */
    sparse_insert(A, n - 6, n - 6, 16e-4);
    sparse_insert(A, n - 7, n - 7, 8e-4);
    sparse_insert(A, n - 8, n - 8, 4e-4);
    sparse_insert(A, n - 9, n - 9, 2e-4);
    sparse_insert(A, n - 10, n - 10, 1e-4);
    return A;
}

/* n = 200, k = 5 largest, matches the Day 10 plan fixture.
 * The top-5 eigenvalues (100, 200, 400, 800, 1600) are
 * well-separated by factor-of-2 gaps, so Lanczos converges to
 * them quickly (<< 100 iterations).  Analytic reference comes
 * from the fixture's construction. */
static void test_s20_day10_eigs_sym_largest(void) {
    const idx_t n = 200;
    const idx_t k = 5;
    SparseMatrix *A = s20_day10_build_gapped_spd(n);
    ASSERT_NOT_NULL(A);

    double vals[5] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .max_iterations = 100,
        .tol = 1e-10,
        .reorthogonalize = 1,
    };

    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &result));
    ASSERT_EQ(result.n_requested, k);
    ASSERT_EQ(result.n_converged, k);
    ASSERT_TRUE(result.iterations < 100);

    /* Descending order (LARGEST convention). */
    const double ref_largest[5] = {1600.0, 800.0, 400.0, 200.0, 100.0};
    for (idx_t j = 0; j < k; j++) {
        double rel = fabs(result.eigenvalues[j] - ref_largest[j]) / fabs(ref_largest[j]);
        ASSERT_TRUE(rel < 1e-10);
    }

    sparse_free(A);
}

/* SMALLEST on a worst-case fixture where Lanczos's slow
 * convergence rate (gap ratio ≈ small-gap / spectrum-scale) is a
 * fundamental algorithmic limitation.  Shift-invert (Day 12) is
 * the standard cure.  Day 10 exercises the SMALLEST branch on a
 * small 2-element diagonal fixture where convergence is trivial —
 * the branch's code path (ascending output, first-k selection) is
 * verified; rigorous SMALLEST testing is deferred to Day 13
 * alongside the shift-invert-capable reference comparisons. */
static void test_s20_day10_eigs_sym_smallest_trivial(void) {
    const idx_t n = 3;
    const idx_t k = 1;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    sparse_insert(A, 0, 0, 1.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 3.0);

    double vals[1] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_SMALLEST,
        .max_iterations = 100,
        .tol = 1e-10,
        .reorthogonalize = 1,
    };

    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &result));
    ASSERT_EQ(result.n_converged, k);
    ASSERT_NEAR(result.eigenvalues[0], 1.0, 1e-10);

    sparse_free(A);
}

/* Wide-spectrum diagonal (no gap structure) — exercises the
 * outer-loop restart logic on a harder problem.  Top-3 are the
 * three largest exponentials on a log-spaced diagonal. */
static void test_s20_day10_eigs_sym_wide_spectrum(void) {
    const idx_t n = 120;
    const idx_t k = 3;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    for (idx_t i = 0; i < n; i++) {
        double s = pow(10.0, 4.0 * (double)i / (double)(n - 1));
        sparse_insert(A, i, i, s);
    }

    double vals[3] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .max_iterations = 200,
        .tol = 1e-10,
        .reorthogonalize = 1,
    };

    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &result));
    ASSERT_EQ(result.n_converged, k);

    /* Reference: diagonal matrix — eigenvalues are the diagonal
     * entries.  Top 3 are A[n-1, n-1], A[n-2, n-2], A[n-3, n-3]. */
    for (idx_t j = 0; j < k; j++) {
        double ref = pow(10.0, 4.0 * (double)(n - 1 - j) / (double)(n - 1));
        double rel = fabs(result.eigenvalues[j] - ref) / fabs(ref);
        ASSERT_TRUE(rel < 1e-10);
    }

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
    RUN_TEST(test_s20_eigs_well_formed_call_succeeds);
    RUN_TEST(test_s20_eigs_null_opts_uses_defaults);
    RUN_TEST(test_s20_eigs_compute_vectors_rejected_day10);
    RUN_TEST(test_s20_eigs_nearest_sigma_rejected_day10);

    /* Day 8 — Lanczos 3-term recurrence */
    RUN_TEST(test_s20_day8_lanczos_diagonal_spectrum);
    RUN_TEST(test_s20_day8_lanczos_tridiagonal_identity);
    RUN_TEST(test_s20_day8_lanczos_rejects_bad_args);

    /* Day 9 — Lanczos full reorthogonalization */
    RUN_TEST(test_s20_day9_reorth_maintains_orthogonality);
    RUN_TEST(test_s20_day9_no_reorth_completes_cleanly);
    RUN_TEST(test_s20_day9_reorth_agrees_with_basic_on_small_matrix);

    /* Day 10 — thick-restart Lanczos + sparse_eigs_sym end-to-end */
    RUN_TEST(test_s20_day10_eigs_sym_largest);
    RUN_TEST(test_s20_day10_eigs_sym_smallest_trivial);
    RUN_TEST(test_s20_day10_eigs_sym_wide_spectrum);

    TEST_SUITE_END();
}
