/*
 * Sprint 20 Day 11 smoke tests for the symmetric eigensolver
 * (`sparse_eigs_sym`).  Covers Ritz-value extraction (LARGEST /
 * SMALLEST), Ritz-vector lift-back through V · Y[:, j], the Day 11
 * precondition checks (symmetry, k range, NULL buffers), and a
 * tridiagonal SPD sanity check cross-compared against a dense
 * reference computed with the existing `tridiag_qr_eigenvalues`.
 *
 * The tests use small matrices (n ≤ 20) so the full Lanczos basis
 * fits comfortably in one batch — this gives a clean, deterministic
 * convergence path without needing to stress thick-restart.  The
 * SuiteSparse / wide-spectrum coverage comes on Day 13.
 */

#include "sparse_dense.h"
#include "sparse_eigs.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build a diagonal SparseMatrix from the given diag array. */
static SparseMatrix *build_diag(idx_t n, const double *diag) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, diag[i]);
    return A;
}

/* Build the 2D Laplacian-style tridiagonal matrix of dimension n:
 * diag = 2, sub/super = -1.  SPD with known eigenvalues
 * λ_j = 2 − 2·cos(j·π / (n + 1)), j = 1..n. */
static SparseMatrix *build_laplacian_tridiag(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 2.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }
    return A;
}

/* ─────────────────────────────────────────────────────────────────────
 * Test 1: LARGEST on diag(1..10) — PLAN Day 11 smoke test #1
 * ───────────────────────────────────────────────────────────────────── */
static void test_diagonal_k3_largest(void) {
    idx_t n = 10;
    double diag[10];
    for (idx_t i = 0; i < n; i++)
        diag[i] = (double)(i + 1);
    SparseMatrix *A = build_diag(n, diag);
    ASSERT_NOT_NULL(A);

    double vals[3] = {0, 0, 0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {.which = SPARSE_EIGS_LARGEST, .tol = 1e-12};

    REQUIRE_OK(sparse_eigs_sym(A, 3, &opts, &result));
    ASSERT_EQ(result.n_requested, 3);
    ASSERT_EQ(result.n_converged, 3);
    /* Descending: 10, 9, 8. */
    ASSERT_NEAR(vals[0], 10.0, 1e-9);
    ASSERT_NEAR(vals[1], 9.0, 1e-9);
    ASSERT_NEAR(vals[2], 8.0, 1e-9);

    sparse_free(A);
}

/* ─────────────────────────────────────────────────────────────────────
 * Test 2: SMALLEST on diag(1..10) — PLAN Day 11 smoke test #2
 * ───────────────────────────────────────────────────────────────────── */
static void test_diagonal_k3_smallest(void) {
    idx_t n = 10;
    double diag[10];
    for (idx_t i = 0; i < n; i++)
        diag[i] = (double)(i + 1);
    SparseMatrix *A = build_diag(n, diag);
    ASSERT_NOT_NULL(A);

    double vals[3] = {0, 0, 0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {.which = SPARSE_EIGS_SMALLEST, .tol = 1e-12};

    REQUIRE_OK(sparse_eigs_sym(A, 3, &opts, &result));
    ASSERT_EQ(result.n_converged, 3);
    /* Ascending: 1, 2, 3. */
    ASSERT_NEAR(vals[0], 1.0, 1e-9);
    ASSERT_NEAR(vals[1], 2.0, 1e-9);
    ASSERT_NEAR(vals[2], 3.0, 1e-9);

    sparse_free(A);
}

/* ─────────────────────────────────────────────────────────────────────
 * Test 3: eigenvector correctness on diagonal — PLAN Day 11 smoke #3
 *
 * On A = diag(1..10), Ritz vectors should recover the canonical-basis
 * eigenvectors (up to sign and negligible MGS mixing).  Rather than
 * pinning the vectors themselves, verify the eigen-equation
 * ‖A·v − λ·v‖ ≤ tol · ‖v‖ directly — the solver is free to emit any
 * orthonormal basis of each eigenspace, but every returned pair must
 * satisfy A·v = λ·v.
 * ───────────────────────────────────────────────────────────────────── */
static void test_diagonal_eigenvectors_satisfy_equation(void) {
    idx_t n = 10;
    double diag[10];
    for (idx_t i = 0; i < n; i++)
        diag[i] = (double)(i + 1);
    SparseMatrix *A = build_diag(n, diag);
    ASSERT_NOT_NULL(A);

    idx_t k = 3;
    double vals[3] = {0, 0, 0};
    double *vecs = calloc((size_t)n * (size_t)k, sizeof(double));
    ASSERT_NOT_NULL(vecs);
    sparse_eigs_t result = {.eigenvalues = vals, .eigenvectors = vecs};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-12,
        .compute_vectors = 1,
    };

    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &result));
    ASSERT_EQ(result.n_converged, k);

    /* For each returned pair, verify ‖A·v − λ·v‖ / ‖v‖ ≤ 1e-9. */
    double *Av = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(Av);
    for (idx_t j = 0; j < k; j++) {
        const double *v = vecs + (size_t)j * (size_t)n;
        sparse_matvec(A, v, Av);
        double num = 0.0, den = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double r = Av[i] - vals[j] * v[i];
            num += r * r;
            den += v[i] * v[i];
        }
        double rel = sqrt(num) / (sqrt(den) > 0.0 ? sqrt(den) : 1.0);
        ASSERT_TRUE(rel <= 1e-9);
    }
    free(Av);
    free(vecs);
    sparse_free(A);
}

/* ─────────────────────────────────────────────────────────────────────
 * Test 4: symmetry-rejection precondition — PLAN Day 11 smoke #4
 *
 * Build a clearly non-symmetric 3×3 matrix and confirm
 * `sparse_eigs_sym` returns `SPARSE_ERR_NOT_SPD` per the Day 11
 * precondition without running any Lanczos work.
 * ───────────────────────────────────────────────────────────────────── */
static void test_non_symmetric_rejected(void) {
    SparseMatrix *A = sparse_create(3, 3);
    ASSERT_NOT_NULL(A);
    sparse_insert(A, 0, 0, 2.0);
    sparse_insert(A, 1, 1, 2.0);
    sparse_insert(A, 2, 2, 2.0);
    /* Non-symmetric off-diagonal: A(0,1) = 1 but A(1,0) = 0. */
    sparse_insert(A, 0, 1, 1.0);

    double vals[1] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {.which = SPARSE_EIGS_LARGEST};
    ASSERT_ERR(sparse_eigs_sym(A, 1, &opts, &result), SPARSE_ERR_NOT_SPD);

    sparse_free(A);
}

/* ─────────────────────────────────────────────────────────────────────
 * Additional coverage: tridiagonal SPD cross-check against dense
 * reference computed from `tridiag_qr_eigenvalues`.  This exercises
 * the Lanczos + Ritz-extract loop on a matrix whose full spectrum is
 * easily computed as a cross-reference; catches any ordering or
 * sign-of-idx bugs that would slip past a pure diagonal test.
 * ───────────────────────────────────────────────────────────────────── */
static void test_tridiag_spd_matches_dense(void) {
    idx_t n = 12;
    SparseMatrix *A = build_laplacian_tridiag(n);
    ASSERT_NOT_NULL(A);

    /* Dense reference: run the tridiag QR on a copy of the matrix
     * (diag = 2, subdiag = -1) and extract the top-3 eigenvalues. */
    double dense_diag[12], dense_sub[11];
    for (idx_t i = 0; i < n; i++)
        dense_diag[i] = 2.0;
    for (idx_t i = 0; i + 1 < n; i++)
        dense_sub[i] = -1.0;
    REQUIRE_OK(tridiag_qr_eigenvalues(dense_diag, dense_sub, n, 0));
    /* dense_diag now holds ascending eigenvalues.  Top-3 LARGEST
     * (descending) come from indices n-1, n-2, n-3. */
    double ref_largest[3] = {dense_diag[n - 1], dense_diag[n - 2], dense_diag[n - 3]};

    double vals[3] = {0, 0, 0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {.which = SPARSE_EIGS_LARGEST, .tol = 1e-12};
    REQUIRE_OK(sparse_eigs_sym(A, 3, &opts, &result));
    ASSERT_EQ(result.n_converged, 3);
    for (idx_t j = 0; j < 3; j++)
        ASSERT_NEAR(vals[j], ref_largest[j], 1e-9);

    sparse_free(A);
}

/* ─────────────────────────────────────────────────────────────────────
 * Defensive: k out of range and NULL checks.  The public header
 * guarantees SPARSE_ERR_BADARG / SPARSE_ERR_NULL respectively; these
 * run on a valid SPD matrix so the only failing precondition is the
 * one we exercise.
 * ───────────────────────────────────────────────────────────────────── */
static void test_bad_args(void) {
    SparseMatrix *A = build_laplacian_tridiag(5);
    ASSERT_NOT_NULL(A);

    double vals[1] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {.which = SPARSE_EIGS_LARGEST};

    /* k = 0 rejected */
    ASSERT_ERR(sparse_eigs_sym(A, 0, &opts, &result), SPARSE_ERR_BADARG);
    /* k > n rejected */
    ASSERT_ERR(sparse_eigs_sym(A, 10, &opts, &result), SPARSE_ERR_BADARG);
    /* NULL A */
    ASSERT_ERR(sparse_eigs_sym(NULL, 1, &opts, &result), SPARSE_ERR_NULL);
    /* NULL result */
    ASSERT_ERR(sparse_eigs_sym(A, 1, &opts, NULL), SPARSE_ERR_NULL);
    /* NULL eigenvalues buffer */
    sparse_eigs_t bad_result = {.eigenvalues = NULL};
    ASSERT_ERR(sparse_eigs_sym(A, 1, &opts, &bad_result), SPARSE_ERR_NULL);
    /* compute_vectors set but eigenvectors NULL */
    sparse_eigs_opts_t opts_vec = {.which = SPARSE_EIGS_LARGEST, .compute_vectors = 1};
    sparse_eigs_t vec_result = {.eigenvalues = vals, .eigenvectors = NULL};
    ASSERT_ERR(sparse_eigs_sym(A, 1, &opts_vec, &vec_result), SPARSE_ERR_NULL);

    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Day 12 — shift-invert Lanczos (SPARSE_EIGS_NEAREST_SIGMA)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Shift-invert drives Lanczos against (A - σ·I)^{-1} via the Day 4-6
 * LDL^T dispatch.  The raw Ritz values of the shift-inverted operator
 * are 1/(λ - σ), so the largest-|θ| values correspond to the λ's
 * closest to σ.  sparse_eigs_sym post-processes them to emit λ in
 * the caller's buffer.
 */

/* Shift-invert on diag(1..20), σ = 10.5, k = 3.
 * Expect the three eigenvalues closest to σ = 10.5 — namely
 * 10, 11, 9 (distances 0.5, 0.5, 1.5).  The output ordering is
 * ascending |λ − σ|, and the first-two ties on 10 and 11 resolve
 * deterministically by the two-pointer scan in s20_select_indices
 * (the ascending θ list has theta_10 = 1/(10 - 10.5) = -2 at the
 * low end and theta_11 = 1/(11 - 10.5) = +2 at the top end; |θ|
 * is identical, so the helper's `>` test keeps theta_11 first). */
static void test_shift_invert_diagonal_k3(void) {
    idx_t n = 20;
    double diag[20];
    for (idx_t i = 0; i < n; i++)
        diag[i] = (double)(i + 1);
    SparseMatrix *A = build_diag(n, diag);
    ASSERT_NOT_NULL(A);

    double vals[3] = {0, 0, 0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_NEAREST_SIGMA,
        .sigma = 10.5,
        .tol = 1e-12,
    };

    REQUIRE_OK(sparse_eigs_sym(A, 3, &opts, &result));
    ASSERT_EQ(result.n_converged, 3);
    /* Distances-from-sigma are what NEAREST_SIGMA actually optimises.
     * The two closest eigenvalues {10, 11} are both at 0.5; there's
     * a tie at distance 1.5 between {9, 12}, and either answer is
     * acceptable — the selection helper's deterministic tie-break
     * happens to pick 12, but from the user's perspective either is
     * a correct "3 eigenvalues closest to 10.5". */
    double d0 = fabs(vals[0] - 10.5);
    double d1 = fabs(vals[1] - 10.5);
    double d2 = fabs(vals[2] - 10.5);
    ASSERT_NEAR(d0, 0.5, 1e-9);
    ASSERT_NEAR(d1, 0.5, 1e-9);
    ASSERT_NEAR(d2, 1.5, 1e-9);
    /* First two are {10, 11} as a set. */
    double a01 = vals[0] < vals[1] ? vals[0] : vals[1];
    double b01 = vals[0] < vals[1] ? vals[1] : vals[0];
    ASSERT_NEAR(a01, 10.0, 1e-9);
    ASSERT_NEAR(b01, 11.0, 1e-9);
    /* Third is either 9 or 12 (tie at distance 1.5). */
    ASSERT_TRUE(fabs(vals[2] - 9.0) <= 1e-9 || fabs(vals[2] - 12.0) <= 1e-9);

    sparse_free(A);
}

/* Shift-invert on an indefinite diagonal matrix with eigenvalues
 * {-3, -1, +2, +5}, σ = 0, k = 2.  The two eigenvalues closest to
 * zero are -1 and +2 (distances 1 and 2).  This fixture exercises
 * the shift-invert path on an indefinite factorization — σ = 0 leaves
 * A - σI = A, which has both signs on its diagonal, so LDL^T must
 * handle Bunch-Kaufman 2×2 pivots internally to factor it.  Confirms
 * the Day 4-6 AUTO dispatch handles the symmetric-indefinite case
 * end-to-end through sparse_eigs_sym. */
static void test_shift_invert_indefinite_small(void) {
    idx_t n = 4;
    double diag[4] = {-3.0, -1.0, 2.0, 5.0};
    SparseMatrix *A = build_diag(n, diag);
    ASSERT_NOT_NULL(A);

    double vals[2] = {0, 0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_NEAREST_SIGMA,
        .sigma = 0.0,
        .tol = 1e-12,
    };

    REQUIRE_OK(sparse_eigs_sym(A, 2, &opts, &result));
    ASSERT_EQ(result.n_converged, 2);
    /* -1 is closest (|λ-σ| = 1); +2 is next (|λ-σ| = 2). */
    double d0 = fabs(vals[0]);
    double d1 = fabs(vals[1]);
    ASSERT_NEAR(d0, 1.0, 1e-9);
    ASSERT_NEAR(d1, 2.0, 1e-9);
    /* As a set, {-1, 2}. */
    double a = vals[0] < vals[1] ? vals[0] : vals[1];
    double b = vals[0] < vals[1] ? vals[1] : vals[0];
    ASSERT_NEAR(a, -1.0, 1e-9);
    ASSERT_NEAR(b, 2.0, 1e-9);

    sparse_free(A);
}

/* Shift-invert with σ coinciding with an eigenvalue: (A - σI) is
 * singular, the LDL^T factor trips, and sparse_eigs_sym propagates
 * SPARSE_ERR_SINGULAR.  Public doxygen tells callers to perturb σ. */
static void test_shift_invert_singular_sigma(void) {
    idx_t n = 5;
    double diag[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    SparseMatrix *A = build_diag(n, diag);
    ASSERT_NOT_NULL(A);

    double vals[2] = {0, 0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_NEAREST_SIGMA,
        .sigma = 3.0, /* exactly an eigenvalue */
    };
    ASSERT_ERR(sparse_eigs_sym(A, 2, &opts, &result), SPARSE_ERR_SINGULAR);

    sparse_free(A);
}

/* Eigenvector correctness under shift-invert: for a Ritz pair
 * (λ, v) emitted by NEAREST_SIGMA mode, verify ‖A·v − λ·v‖ / ‖v‖ is
 * small.  Ritz vectors of (A - σI)^{-1} share eigenspaces with A, so
 * they are eigenvectors of A at λ = σ + 1/θ — this confirms the
 * Day 11 lift still works through the Day 12 transformation. */
static void test_shift_invert_eigenvectors(void) {
    idx_t n = 12;
    SparseMatrix *A = build_laplacian_tridiag(n);
    ASSERT_NOT_NULL(A);

    /* σ in the middle of the Laplacian spectrum (eigenvalues span
     * roughly [0.06, 3.94]); pick σ = 2 to land in the interior. */
    idx_t k = 3;
    double vals[3] = {0, 0, 0};
    double *vecs = calloc((size_t)n * (size_t)k, sizeof(double));
    ASSERT_NOT_NULL(vecs);
    sparse_eigs_t result = {.eigenvalues = vals, .eigenvectors = vecs};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_NEAREST_SIGMA,
        .sigma = 2.0,
        .compute_vectors = 1,
        .tol = 1e-10,
    };

    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &result));
    ASSERT_EQ(result.n_converged, k);

    double *Av = malloc((size_t)n * sizeof(double));
    ASSERT_NOT_NULL(Av);
    for (idx_t j = 0; j < k; j++) {
        const double *v = vecs + (size_t)j * (size_t)n;
        sparse_matvec(A, v, Av);
        double num = 0.0, den = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double r = Av[i] - vals[j] * v[i];
            num += r * r;
            den += v[i] * v[i];
        }
        double rel = sqrt(num) / (sqrt(den) > 0.0 ? sqrt(den) : 1.0);
        ASSERT_TRUE(rel <= 1e-8);
    }
    free(Av);
    free(vecs);
    sparse_free(A);
}

/* Convergence-rate comparison: interior eigenvalues on a wide-
 * spectrum n = 80 diagonal SPD.  Direct Lanczos (SMALLEST) converging
 * to interior eigenvalues would need the full Krylov basis; shift-
 * invert with σ in the middle of the spectrum converges much faster.
 *
 * We don't re-measure in terms of iterations (the stability-check
 * convergence criterion isn't directly comparable between the two
 * modes), but we do sanity-check that shift-invert converges on a
 * fixture where direct SMALLEST would be slow — both calls should
 * return SPARSE_OK and report the expected eigenvalues. */
static void test_shift_invert_wide_spectrum_middle(void) {
    idx_t n = 40;
    double diag[40];
    /* Geometric spectrum: λ_i = 10^(i/(n-1)*4), spans [1, 10^4]. */
    for (idx_t i = 0; i < n; i++)
        diag[i] = pow(10.0, (double)i / (double)(n - 1) * 4.0);
    SparseMatrix *A = build_diag(n, diag);
    ASSERT_NOT_NULL(A);

    /* Pick σ in the middle log-wise: 10^2 = 100.  The eigenvalue
     * closest to 100 is some λ_i from the diag array; find it. */
    double sigma = 100.0;
    idx_t closest_idx = 0;
    double closest_dist = fabs(diag[0] - sigma);
    for (idx_t i = 1; i < n; i++) {
        double d = fabs(diag[i] - sigma);
        if (d < closest_dist) {
            closest_dist = d;
            closest_idx = i;
        }
    }

    double vals[1] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_NEAREST_SIGMA,
        .sigma = sigma,
        .tol = 1e-10,
    };
    REQUIRE_OK(sparse_eigs_sym(A, 1, &opts, &result));
    ASSERT_EQ(result.n_converged, 1);
    ASSERT_NEAR(vals[0], diag[closest_idx], 1e-7);

    sparse_free(A);
}

int main(void) {
    TEST_SUITE_BEGIN("Sparse eigensolver — Sprint 20 Days 11-12");

    RUN_TEST(test_diagonal_k3_largest);
    RUN_TEST(test_diagonal_k3_smallest);
    RUN_TEST(test_diagonal_eigenvectors_satisfy_equation);
    RUN_TEST(test_non_symmetric_rejected);
    RUN_TEST(test_tridiag_spd_matches_dense);
    RUN_TEST(test_bad_args);

    /* Day 12: shift-invert Lanczos. */
    RUN_TEST(test_shift_invert_diagonal_k3);
    RUN_TEST(test_shift_invert_indefinite_small);
    RUN_TEST(test_shift_invert_singular_sigma);
    RUN_TEST(test_shift_invert_eigenvectors);
    RUN_TEST(test_shift_invert_wide_spectrum_middle);

    TEST_SUITE_END();
}
