/*
 * Sprint 21 thick-restart Lanczos tests.
 *
 * Day 2 scope (this file's initial contents):
 *   - Arrowhead-to-tridiagonal reduction round-trip: the reduced
 *     tridiagonal has the same spectrum as the original arrowhead
 *     to 1e-12.  Reference spectrum computed via a simple dense
 *     Jacobi rotation solver implemented in this file (not
 *     production-quality; sized for the small test matrices).
 *   - `lanczos_restart_pick_locked` output correctness: run a
 *     small Lanczos with full reorth, select k Ritz pairs, pick
 *     the locked block; verify V_locked^T V_locked ≈ I to 1e-10
 *     (Sprint 20 Day 14 documented that `reorthogonalize = 1` is
 *     required for this invariant — exercised here).
 *   - `lanczos_restart_state_assemble` round-trip: pack into a
 *     state; verify fields; round-trip through free.
 *
 * Days 3, 4, 12 extend this file with phase-execution tests,
 * memory-bounded convergence, and the full thick-restart
 * regression corpus per SPRINT_21/PLAN.md.
 */

#include "sparse_eigs.h"
#include "sparse_eigs_internal.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"
#include "test_framework.h"

#include "sparse_dense.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── Dense symmetric Jacobi eigenvalue reference (test-only) ─────── */

/* Classical Jacobi rotation sweep for dense symmetric eigenvalues.
 * Not production quality: O(K^3) per sweep, ~O(log K) sweeps,
 * totally acceptable for the K ≤ 20 matrices exercised here.  Used
 * as an independent reference against which the arrowhead reduction
 * + tridiag_qr_eigenvalues spectrum is checked. */
static void test_dense_sym_eigvals_jacobi(double *A, idx_t K, double *eigvals) {
    const idx_t max_sweeps = 100;
    const double tol = 1e-14;
    for (idx_t sweep = 0; sweep < max_sweeps; sweep++) {
        /* off-diagonal Frobenius norm */
        double off = 0.0;
        for (idx_t i = 0; i < K; i++)
            for (idx_t j = i + 1; j < K; j++)
                off += A[i + j * K] * A[i + j * K];
        if (sqrt(off) < tol)
            break;
        for (idx_t p = 0; p < K; p++) {
            for (idx_t q = p + 1; q < K; q++) {
                double apq = A[p + q * K];
                if (fabs(apq) < tol)
                    continue;
                double app = A[p + p * K];
                double aqq = A[q + q * K];
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (fabs(theta) > 1e15)
                    t = 1.0 / (2.0 * theta);
                else {
                    double sign_t = theta >= 0.0 ? 1.0 : -1.0;
                    t = sign_t / (fabs(theta) + sqrt(theta * theta + 1.0));
                }
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;
                /* Apply the 2×2 rotation to rows and cols p, q. */
                for (idx_t i = 0; i < K; i++) {
                    if (i == p || i == q)
                        continue;
                    double aip = A[i + p * K];
                    double aiq = A[i + q * K];
                    A[i + p * K] = c * aip - s * aiq;
                    A[p + i * K] = A[i + p * K];
                    A[i + q * K] = s * aip + c * aiq;
                    A[q + i * K] = A[i + q * K];
                }
                A[p + p * K] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                A[q + q * K] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                A[p + q * K] = 0.0;
                A[q + p * K] = 0.0;
            }
        }
    }
    for (idx_t i = 0; i < K; i++)
        eigvals[i] = A[i + i * K];
}

static int cmp_doubles_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db)
        return -1;
    if (da > db)
        return 1;
    return 0;
}

/* ─── Test 1: arrowhead reduction round-trip (spectrum preserved) ── */

/* Build a k_locked = 3, m_ext = 4 arrowhead with hand-picked entries.
 * Run s21_arrowhead_to_tridiag; compare the reduced spectrum against
 * the reference dense-Jacobi spectrum to 1e-12. */
static void test_arrowhead_to_tridiag_preserves_spectrum(void) {
    const idx_t k_locked = 3;
    const idx_t m_ext = 4;
    const idx_t K = k_locked + m_ext;

    double theta_locked[3] = {1.0, 2.5, 4.0};
    double beta_coupling[3] = {0.7, -0.5, 0.3};
    double alpha_ext[4] = {3.0, 2.0, 5.0, 1.5};
    double beta_ext[3] = {0.4, 0.6, -0.2};

    double diag[7];
    double subdiag[6];
    REQUIRE_OK(s21_arrowhead_to_tridiag(theta_locked, beta_coupling, k_locked, alpha_ext, beta_ext,
                                        m_ext, diag, subdiag));

    /* Reference: materialise the same arrowhead as a dense K×K
     * symmetric matrix and compute eigenvalues via Jacobi. */
    double *A_ref = calloc((size_t)K * (size_t)K, sizeof(double));
    ASSERT_NOT_NULL(A_ref);
    if (!A_ref)
        return;
    for (idx_t i = 0; i < k_locked; i++)
        A_ref[i + i * K] = theta_locked[i];
    for (idx_t i = 0; i < m_ext; i++)
        A_ref[(k_locked + i) + (k_locked + i) * K] = alpha_ext[i];
    for (idx_t j = 0; j < k_locked; j++) {
        A_ref[k_locked + j * K] = beta_coupling[j];
        A_ref[j + k_locked * K] = beta_coupling[j];
    }
    for (idx_t i = 0; i + 1 < m_ext; i++) {
        idx_t r = k_locked + i;
        A_ref[(r + 1) + r * K] = beta_ext[i];
        A_ref[r + (r + 1) * K] = beta_ext[i];
    }
    double ref_eigs[7];
    test_dense_sym_eigvals_jacobi(A_ref, K, ref_eigs);
    qsort(ref_eigs, (size_t)K, sizeof(double), cmp_doubles_asc);
    free(A_ref);

    /* Reduced tridiagonal eigenvalues via tridiag_qr_eigenvalues.
     * Caller pays the destructive call's data copy. */
    double diag_copy[7];
    double subdiag_copy[6];
    memcpy(diag_copy, diag, sizeof(diag_copy));
    memcpy(subdiag_copy, subdiag, sizeof(subdiag_copy));
    REQUIRE_OK(tridiag_qr_eigenvalues(diag_copy, subdiag_copy, K, 0));
    /* tridiag_qr_eigenvalues returns ascending eigenvalues in diag. */

    for (idx_t i = 0; i < K; i++)
        ASSERT_NEAR(diag_copy[i], ref_eigs[i], 1e-12);
}

/* Edge case: k_locked = 1 reduces to just the subdiagonal at
 * position 0 being beta_coupling[0], and the rest already
 * tridiagonal.  Verifies the helper accepts the degenerate case. */
static void test_arrowhead_to_tridiag_k_locked_one(void) {
    const idx_t k_locked = 1;
    const idx_t m_ext = 3;
    const idx_t K = k_locked + m_ext;
    double theta_locked[1] = {5.0};
    double beta_coupling[1] = {0.9};
    double alpha_ext[3] = {2.0, 3.0, 4.0};
    double beta_ext[2] = {0.5, 0.7};
    double diag[4];
    double subdiag[3];
    REQUIRE_OK(s21_arrowhead_to_tridiag(theta_locked, beta_coupling, k_locked, alpha_ext, beta_ext,
                                        m_ext, diag, subdiag));
    /* k_locked == 1 means the arrowhead is already tridiagonal;
     * the reduction should be the identity (straight copy). */
    ASSERT_NEAR(diag[0], 5.0, 1e-14);
    ASSERT_NEAR(diag[1], 2.0, 1e-14);
    ASSERT_NEAR(diag[2], 3.0, 1e-14);
    ASSERT_NEAR(diag[3], 4.0, 1e-14);
    ASSERT_NEAR(subdiag[0], 0.9, 1e-14);
    ASSERT_NEAR(subdiag[1], 0.5, 1e-14);
    ASSERT_NEAR(subdiag[2], 0.7, 1e-14);

    /* Double-check spectrum via QR. */
    double diag_copy[4];
    double subdiag_copy[3];
    memcpy(diag_copy, diag, sizeof(diag_copy));
    memcpy(subdiag_copy, subdiag, sizeof(subdiag_copy));
    REQUIRE_OK(tridiag_qr_eigenvalues(diag_copy, subdiag_copy, K, 0));
    /* All eigenvalues real and bounded — just a sanity smoke. */
    for (idx_t i = 0; i < K; i++)
        ASSERT_TRUE(diag_copy[i] > 0.0 && diag_copy[i] < 10.0);
}

/* ─── Test 2: pick_locked forms orthonormal V_locked ─────────────── */

/* Given a completed Lanczos run on a small diagonal SPD, pick k=3
 * locked pairs and verify V_locked^T V_locked ≈ I (orthonormality
 * inherited from the orthonormal V via the orthonormal Y). */
static void test_pick_locked_orthonormal(void) {
    /* Build a well-conditioned diagonal SPD A = diag(1..6). */
    const idx_t n = 6;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    /* Run Lanczos with full reorth to completion (m = n = 6). */
    double v0[6] = {1.0, 0.5, -0.3, 0.7, 0.2, -0.9};
    double V[36];
    double alpha[6];
    double beta[6];
    idx_t m_actual = 0;
    REQUIRE_OK(lanczos_iterate(A, v0, n, /*reorthogonalize=*/1, V, alpha, beta, &m_actual));
    ASSERT_EQ(m_actual, n);

    /* Ritz extraction: theta, Y. */
    double theta[6];
    double Y[36];
    double subdiag_scratch[6];
    for (idx_t i = 0; i < n; i++)
        theta[i] = alpha[i];
    for (idx_t i = 0; i + 1 < n; i++)
        subdiag_scratch[i] = beta[i];
    REQUIRE_OK(tridiag_qr_eigenpairs(theta, subdiag_scratch, Y, n, 0));

    /* Select k = 3 largest Ritz pairs (indices n-1, n-2, n-3). */
    const idx_t k = 3;
    idx_t sel_idx[3] = {n - 1, n - 2, n - 3};
    double V_locked[18];       /* n × k */
    double theta_locked[3];
    double beta_coupling[3];
    double beta_m = beta[m_actual - 1];

    lanczos_restart_pick_locked(V, n, m_actual, Y, theta, sel_idx, k, beta_m, V_locked,
                                theta_locked, beta_coupling);

    /* theta_locked should be A's three largest eigenvalues (6, 5, 4)
     * up to Lanczos convergence tolerance. */
    ASSERT_NEAR(theta_locked[0], 6.0, 1e-10);
    ASSERT_NEAR(theta_locked[1], 5.0, 1e-10);
    ASSERT_NEAR(theta_locked[2], 4.0, 1e-10);

    /* V_locked columns should be orthonormal: V_locked^T V_locked = I. */
    for (idx_t i = 0; i < k; i++) {
        for (idx_t j = 0; j < k; j++) {
            double dot = 0.0;
            for (idx_t r = 0; r < n; r++)
                dot += V_locked[r + i * n] * V_locked[r + j * n];
            double expect = (i == j) ? 1.0 : 0.0;
            ASSERT_NEAR(dot, expect, 1e-10);
        }
    }

    /* V_locked[:, j] should be an eigenvector of A with eigenvalue
     * theta_locked[j].  Check ||A v - theta v|| ≤ 1e-10 per pair. */
    double Av[6];
    for (idx_t j = 0; j < k; j++) {
        sparse_matvec(A, V_locked + j * n, Av);
        double res = 0.0;
        for (idx_t r = 0; r < n; r++) {
            double delta = Av[r] - theta_locked[j] * V_locked[r + j * n];
            res += delta * delta;
        }
        ASSERT_TRUE(sqrt(res) < 1e-9);
    }

    sparse_free(A);
}

/* ─── Test 3: state_assemble round-trip ─────────────────────────── */

/* Pack a restart state from synthetic inputs; verify fields; free;
 * assert double-free is safe. */
static void test_restart_state_assemble_roundtrip(void) {
    lanczos_restart_state_t state = {0};
    const idx_t n = 5;
    const idx_t k = 2;
    double V_locked_src[10] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    double theta_locked_src[2] = {7.3, 2.1};
    double beta_coupling_src[2] = {0.5, -0.3};
    double residual_src[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    double residual_norm = sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.4 * 0.4 + 0.5 * 0.5);

    REQUIRE_OK(lanczos_restart_state_assemble(&state, n, k, V_locked_src, theta_locked_src,
                                              beta_coupling_src, residual_src, residual_norm));

    ASSERT_EQ(state.n, n);
    ASSERT_EQ(state.k_locked, k);
    ASSERT_TRUE(state.k_locked_cap >= k);
    ASSERT_NOT_NULL(state.V_locked);
    ASSERT_NOT_NULL(state.theta_locked);
    ASSERT_NOT_NULL(state.beta_coupling);
    ASSERT_NOT_NULL(state.residual);
    ASSERT_NEAR(state.residual_norm, residual_norm, 1e-14);

    /* Buffer contents copied correctly. */
    for (idx_t i = 0; i < n * k; i++)
        ASSERT_NEAR(state.V_locked[i], V_locked_src[i], 1e-14);
    for (idx_t i = 0; i < k; i++) {
        ASSERT_NEAR(state.theta_locked[i], theta_locked_src[i], 1e-14);
        ASSERT_NEAR(state.beta_coupling[i], beta_coupling_src[i], 1e-14);
    }
    for (idx_t i = 0; i < n; i++)
        ASSERT_NEAR(state.residual[i], residual_src[i], 1e-14);

    /* Re-assemble with same k_locked must succeed and reuse buffers
     * (no new allocation expected). */
    idx_t cap_before = state.k_locked_cap;
    double *V_ptr_before = state.V_locked;
    REQUIRE_OK(lanczos_restart_state_assemble(&state, n, k, V_locked_src, theta_locked_src,
                                              beta_coupling_src, residual_src, residual_norm));
    ASSERT_EQ(state.k_locked_cap, cap_before);
    ASSERT_TRUE(state.V_locked == V_ptr_before);

    /* Re-assemble with larger k_locked must grow the cap. */
    const idx_t k2 = 4;
    double V2[20] = {0};
    for (idx_t j = 0; j < k2; j++)
        V2[j + j * n] = 1.0;
    double theta2[4] = {1.0, 2.0, 3.0, 4.0};
    double beta2[4] = {0.1, 0.2, 0.3, 0.4};
    REQUIRE_OK(lanczos_restart_state_assemble(&state, n, k2, V2, theta2, beta2, residual_src,
                                              residual_norm));
    ASSERT_TRUE(state.k_locked_cap >= k2);
    ASSERT_EQ(state.k_locked, k2);
    for (idx_t i = 0; i < k2; i++)
        ASSERT_NEAR(state.theta_locked[i], theta2[i], 1e-14);

    lanczos_restart_state_free(&state);
    ASSERT_EQ(state.k_locked, 0);
    ASSERT_EQ(state.k_locked_cap, 0);
    ASSERT_NULL(state.V_locked);
    ASSERT_NULL(state.theta_locked);
    ASSERT_NULL(state.beta_coupling);
    ASSERT_NULL(state.residual);

    /* Double-free must be safe (re-sets the already-zeroed struct). */
    lanczos_restart_state_free(&state);

    /* NULL pointer is also safe. */
    lanczos_restart_state_free(NULL);
}

/* State with n mismatch must be rejected (can't reuse state across
 * eigenproblems with different dimensions). */
static void test_restart_state_assemble_n_mismatch_rejected(void) {
    lanczos_restart_state_t state = {0};
    const idx_t k = 1;
    double Vs[5] = {1.0, 0.0, 0.0, 0.0, 0.0};
    double theta[1] = {1.0};
    double beta_coupling[1] = {0.5};
    double residual[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    /* First assemble at n=5, then attempt n=7 — should return
     * SPARSE_ERR_SHAPE. */
    REQUIRE_OK(
        lanczos_restart_state_assemble(&state, 5, k, Vs, theta, beta_coupling, residual, 1.0));
    ASSERT_ERR(lanczos_restart_state_assemble(&state, 7, k, Vs, theta, beta_coupling, residual,
                                              1.0),
               SPARSE_ERR_SHAPE);
    lanczos_restart_state_free(&state);
}

/* ═══════════════════════════════════════════════════════════════════
 * Day 3: phase execution + thick-restart outer loop
 * ═══════════════════════════════════════════════════════════════════ */

/* Matvec adapter for the `lanczos_op_fn` callback (the Sprint 20
 * file-scope `s20_op_matvec` helper in `sparse_eigs.c` isn't
 * exported for internal tests). */
static sparse_err_t test_op_matvec(const void *ctx, idx_t n, const double *x, double *y) {
    (void)n;
    return sparse_matvec((const SparseMatrix *)ctx, x, y);
}

/* Empty-state path of `lanczos_thick_restart_iterate`: with a NULL
 * state pointer the iterator must produce bit-for-bit identical
 * output to `lanczos_iterate` on the same fixture.  Exercises the
 * Day 3 fast-path delegation. */
static void test_thick_restart_iterate_empty_state_matches_lanczos(void) {
    const idx_t n = 6;
    const idx_t m = 6;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    double v0[6] = {1.0, 0.5, -0.3, 0.7, 0.2, -0.9};
    double V_ref[36] = {0};
    double alpha_ref[6] = {0};
    double beta_ref[6] = {0};
    idx_t m_ref = 0;
    REQUIRE_OK(lanczos_iterate(A, v0, m, /*reorthogonalize=*/1, V_ref, alpha_ref, beta_ref,
                               &m_ref));

    double V_tr[36] = {0};
    double alpha_tr[6] = {0};
    double beta_tr[6] = {0};
    idx_t m_tr = 0;
    REQUIRE_OK(lanczos_thick_restart_iterate(test_op_matvec, A, n, v0, m, /*reorth=*/1,
                                             /*state=*/NULL, V_tr, alpha_tr, beta_tr, &m_tr));

    ASSERT_EQ(m_ref, m);
    ASSERT_EQ(m_tr, m);
    for (idx_t i = 0; i < n * m; i++)
        ASSERT_NEAR(V_tr[i], V_ref[i], 1e-14);
    for (idx_t i = 0; i < m; i++) {
        ASSERT_NEAR(alpha_tr[i], alpha_ref[i], 1e-14);
        ASSERT_NEAR(beta_tr[i], beta_ref[i], 1e-14);
    }

    sparse_free(A);
}

/* End-to-end convergence on a small SPD diagonal fixture.  The
 * thick-restart backend must produce the same k largest eigenvalues
 * as the grow-m path to 1e-10.  Exercises the full outer-loop
 * pipeline: initial phase + (possibly) one or more restarts +
 * Ritz extraction via dense Jacobi. */
static void test_thick_restart_backend_converges_small(void) {
    const idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    const idx_t k = 3;
    double vals[3] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-10,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &result));
    ASSERT_EQ(result.n_converged, k);
    /* Descending: 10, 9, 8. */
    ASSERT_NEAR(vals[0], 10.0, 1e-9);
    ASSERT_NEAR(vals[1], 9.0, 1e-9);
    ASSERT_NEAR(vals[2], 8.0, 1e-9);

    sparse_free(A);
}

/* SMALLEST branch: exercises the ascending selection path through
 * the thick-restart outer loop. */
static void test_thick_restart_backend_smallest(void) {
    const idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    const idx_t k = 3;
    double vals[3] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_SMALLEST,
        .tol = 1e-10,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &result));
    ASSERT_EQ(result.n_converged, k);
    ASSERT_NEAR(vals[0], 1.0, 1e-9);
    ASSERT_NEAR(vals[1], 2.0, 1e-9);
    ASSERT_NEAR(vals[2], 3.0, 1e-9);

    sparse_free(A);
}

/* Cross-backend parity on a wider spectrum: the thick-restart path
 * should match the grow-m path to 1e-8 on a 2D Laplacian-style
 * tridiagonal SPD (where both converge cleanly). */
static void test_thick_restart_matches_grow_m(void) {
    const idx_t n = 30;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    const idx_t k = 4;
    double vals_grow[4] = {0};
    double vals_tr[4] = {0};

    sparse_eigs_t res_grow = {.eigenvalues = vals_grow};
    sparse_eigs_opts_t opts_grow = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-12,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts_grow, &res_grow));
    ASSERT_EQ(res_grow.n_converged, k);

    sparse_eigs_t res_tr = {.eigenvalues = vals_tr};
    sparse_eigs_opts_t opts_tr = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-12,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts_tr, &res_tr));
    ASSERT_EQ(res_tr.n_converged, k);

    for (idx_t j = 0; j < k; j++)
        ASSERT_NEAR(vals_grow[j], vals_tr[j], 1e-8);

    sparse_free(A);
}

/* compute_vectors = 1 with the thick-restart backend should lift
 * Ritz vectors through V · Y_arrow just like the grow-m path.
 * Verify ||A v_j - theta_j v_j|| ≤ 1e-8 per returned pair. */
static void test_thick_restart_backend_eigenvectors(void) {
    const idx_t n = 10;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)(i + 1));

    const idx_t k = 3;
    double vals[3] = {0};
    double *vecs = calloc((size_t)n * (size_t)k, sizeof(double));
    ASSERT_NOT_NULL(vecs);
    if (!vecs) {
        sparse_free(A);
        return;
    }
    sparse_eigs_t result = {.eigenvalues = vals, .eigenvectors = vecs};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-12,
        .compute_vectors = 1,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &result));
    ASSERT_EQ(result.n_converged, k);

    double Av[10];
    for (idx_t j = 0; j < k; j++) {
        sparse_matvec(A, vecs + (size_t)j * (size_t)n, Av);
        double res = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double delta = Av[i] - vals[j] * vecs[i + j * n];
            res += delta * delta;
        }
        ASSERT_TRUE(sqrt(res) < 1e-8);
    }

    free(vecs);
    sparse_free(A);
}

/* ═══════════════════════════════════════════════════════════════════
 * Day 4: memory-bounded convergence + cross-backend parity
 * ═══════════════════════════════════════════════════════════════════ */

#ifndef DATA_DIR
#define DATA_DIR "tests/data"
#endif
#define SS_DIR DATA_DIR "/suitesparse"

/* bcsstk14 (n=1806) k=5 LARGEST: the memory-bounded regression
 * claim.  Thick-restart at m_restart = 2k + 20 = 30 must
 * converge with `peak_basis_size <= 35 + 2*k = 40` columns
 * regardless of how many restart phases the outer loop runs.
 * That's `40 * 1806 * 8 ≈ 565 KB` of V — roughly 15× smaller
 * than the grow-m path's `500 * 1806 * 8 ≈ 7.2 MB` at the
 * test's `max_iterations = 500` cap.  Assert the bound
 * numerically rather than by measurement. */
static void test_thick_restart_bcsstk14_bounded_memory(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk14.mtx"));
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    idx_t n = sparse_rows(A);
    ASSERT_TRUE(n >= 1800 && n < 2000);

    const idx_t k = 5;
    double *vals = calloc((size_t)k, sizeof(double));
    double *vecs = calloc((size_t)n * (size_t)k, sizeof(double));
    ASSERT_NOT_NULL(vals);
    ASSERT_NOT_NULL(vecs);
    if (!vals || !vecs) {
        free(vals);
        free(vecs);
        sparse_free(A);
        return;
    }

    sparse_eigs_t res = {.eigenvalues = vals, .eigenvectors = vecs};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-8,
        .compute_vectors = 1,
        .max_iterations = 2000,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &res));
    ASSERT_EQ(res.n_converged, k);

    /* Memory bound: peak_basis_size <= m_restart + 2*k = 30 + 10
     * = 40 columns.  With safety margin: <= 45.  This is the
     * Day 4 PROJECT_PLAN target for bounded thick-restart memory. */
    ASSERT_TRUE(res.peak_basis_size <= 45);

    /* Residuals: use the same `||Av - λv|| / (|λ| * ||v||)` scaled
     * formula as Sprint 20's `assert_ritz_residuals` helper —
     * bcsstk14 eigenvalues are ~1e10 so absolute residuals of ~1e3
     * are still relative ~1e-7, consistent with the 1e-8 tol. */
    for (idx_t j = 0; j < k; j++) {
        double *Av = malloc((size_t)n * sizeof(double));
        ASSERT_NOT_NULL(Av);
        if (!Av)
            break;
        sparse_matvec(A, vecs + j * n, Av);
        double res_sq = 0.0, v_sq = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double r = Av[i] - vals[j] * vecs[i + j * n];
            res_sq += r * r;
            v_sq += vecs[i + j * n] * vecs[i + j * n];
        }
        double lambda_abs = fabs(vals[j]);
        double anchor = (lambda_abs > 0.0 ? lambda_abs : 1.0) * (sqrt(v_sq) > 0.0 ? sqrt(v_sq) : 1.0);
        double rel = sqrt(res_sq) / anchor;
        ASSERT_TRUE(rel < 1e-6);
        free(Av);
    }

    free(vals);
    free(vecs);
    sparse_free(A);
}

/* Cross-backend parity helper: run both backends on the same
 * matrix + opts, compare eigenvalues within the given tol.  Used
 * by the four fixture tests below. */
static void s21_day4_parity(SparseMatrix *A, idx_t k, sparse_eigs_which_t which, double sigma,
                            double eig_tol, idx_t max_iter) {
    double *vals_grow = calloc((size_t)k, sizeof(double));
    double *vals_tr = calloc((size_t)k, sizeof(double));
    ASSERT_NOT_NULL(vals_grow);
    ASSERT_NOT_NULL(vals_tr);
    if (!vals_grow || !vals_tr) {
        free(vals_grow);
        free(vals_tr);
        return;
    }

    sparse_eigs_t res_grow = {.eigenvalues = vals_grow};
    sparse_eigs_opts_t opts_grow = {
        .which = which,
        .sigma = sigma,
        .tol = 1e-10,
        .max_iterations = max_iter,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts_grow, &res_grow));
    ASSERT_EQ(res_grow.n_converged, k);

    sparse_eigs_t res_tr = {.eigenvalues = vals_tr};
    sparse_eigs_opts_t opts_tr = {
        .which = which,
        .sigma = sigma,
        .tol = 1e-10,
        .max_iterations = max_iter,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts_tr, &res_tr));
    ASSERT_EQ(res_tr.n_converged, k);

    /* Relative eigenvalue agreement: `eig_tol` is interpreted as a
     * relative tolerance scaled by max(|λ_grow|, 1) so that
     * fixtures with large-magnitude eigenvalues (e.g. bcsstk14
     * with λ ~ 1e10) don't force a brittle absolute comparison. */
    for (idx_t j = 0; j < k; j++) {
        double scale = fabs(vals_grow[j]);
        if (scale < 1.0)
            scale = 1.0;
        double diff = fabs(vals_grow[j] - vals_tr[j]) / scale;
        if (!(diff < eig_tol)) {
            TF_FAIL_("Ritz pair %td: grow=%.15g, tr=%.15g, rel-diff=%.3e > tol=%.3e",
                     (ptrdiff_t)j, vals_grow[j], vals_tr[j], diff, eig_tol);
        }
        tf_asserts++;
    }

    /* Thick-restart peak_basis_size must be strictly smaller than
     * grow-m's on any non-trivial fixture (when max_iterations
     * leaves room for both to exercise their allocation strategy). */
    ASSERT_TRUE(res_tr.peak_basis_size < res_grow.peak_basis_size);

    free(vals_grow);
    free(vals_tr);
}

/* nos4 parity: LARGEST k=5. */
static void test_thick_restart_parity_nos4(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/nos4.mtx"));
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    s21_day4_parity(A, 5, SPARSE_EIGS_LARGEST, 0.0, 1e-8, 300);
    sparse_free(A);
}

/* bcsstk04 parity: LARGEST k=3. */
static void test_thick_restart_parity_bcsstk04(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk04.mtx"));
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    s21_day4_parity(A, 3, SPARSE_EIGS_LARGEST, 0.0, 1e-6, 300);
    sparse_free(A);
}

/* bcsstk14 parity: LARGEST k=5.  Separate from the memory-bound
 * test above — this one asserts eigenvalue agreement between
 * backends. */
static void test_thick_restart_parity_bcsstk14(void) {
    SparseMatrix *A = NULL;
    REQUIRE_OK(sparse_load_mm(&A, SS_DIR "/bcsstk14.mtx"));
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    s21_day4_parity(A, 5, SPARSE_EIGS_LARGEST, 0.0, 1e-5, 2000);
    sparse_free(A);
}

/* AUTO dispatch must route to thick-restart above the threshold.
 * Fabricate a matrix with n > SPARSE_EIGS_THICK_RESTART_THRESHOLD
 * (default 500) and check `peak_basis_size` falls in the
 * thick-restart-bounded regime. */
static void test_thick_restart_auto_dispatch_above_threshold(void) {
    const idx_t n = 600; /* > 500 (default threshold) */
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    /* Diagonal fixture with log-spaced eigenvalues: well-separated
     * so both backends converge in few matvecs.  Avoids the
     * tightly-clustered spectrum of a 2D-Laplacian tridiagonal
     * where thick-restart needs many phases to resolve the top
     * cluster (a convergence-rate property the AUTO dispatch test
     * shouldn't be gated on — that's Day 13's job). */
    for (idx_t i = 0; i < n; i++) {
        double s = pow(10.0, 4.0 * (double)i / (double)(n - 1));
        sparse_insert(A, i, i, s);
    }

    const idx_t k = 3;
    double vals[3] = {0};
    sparse_eigs_t res = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-8,
        .max_iterations = 500,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_AUTO,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &res));
    ASSERT_EQ(res.n_converged, k);

    /* AUTO routed to thick-restart because n >= threshold: peak
     * basis size should be in the thick-restart regime
     * (m_restart + 2k = 2k+20 + 2k = 4k+20 = 32 for k=3) rather
     * than the grow-m regime (m_cap ≈ max(10k+20, 100) = 100). */
    ASSERT_TRUE(res.peak_basis_size <= 50);

    sparse_free(A);
}

/* AUTO dispatch below the threshold must route to grow-m
 * (peak_basis_size > 50 in the grow-m regime where m_cap
 * defaults reach 100).  Complement of the above test. */
static void test_thick_restart_auto_dispatch_below_threshold(void) {
    const idx_t n = 100; /* < 500 */
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++) {
        sparse_insert(A, i, i, 4.0);
        if (i > 0) {
            sparse_insert(A, i, i - 1, -1.0);
            sparse_insert(A, i - 1, i, -1.0);
        }
    }

    const idx_t k = 3;
    double vals[3] = {0};
    sparse_eigs_t res = {.eigenvalues = vals};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-10,
        .reorthogonalize = 1,
        .backend = SPARSE_EIGS_BACKEND_AUTO,
    };
    REQUIRE_OK(sparse_eigs_sym(A, k, &opts, &res));
    ASSERT_EQ(res.n_converged, k);

    /* AUTO routed to grow-m: peak_basis_size ≈ m_cap = max(3k+30, 100)
     * clamped to n → ≥ 100 on this fixture. */
    ASSERT_TRUE(res.peak_basis_size >= 50);

    sparse_free(A);
}

/* Invalid backend rejection — the updated validation at
 * sparse_eigs_sym entry must still accept the new enum value and
 * reject out-of-range ones. */
static void test_thick_restart_backend_bad_enum_rejected(void) {
    const idx_t n = 4;
    SparseMatrix *A = sparse_create(n, n);
    ASSERT_NOT_NULL(A);
    if (!A)
        return;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, 2.0);
    double vals[1] = {0};
    sparse_eigs_t result = {.eigenvalues = vals};
    sparse_eigs_opts_t opts_bad = {
        .backend = (sparse_eigs_backend_t)99,
    };
    ASSERT_ERR(sparse_eigs_sym(A, 1, &opts_bad, &result), SPARSE_ERR_BADARG);
    sparse_free(A);
}

/* ─── Runner ────────────────────────────────────────────────────── */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 21 thick-restart Lanczos — Days 2-3");

    /* Day 2: arrowhead helpers. */
    RUN_TEST(test_arrowhead_to_tridiag_preserves_spectrum);
    RUN_TEST(test_arrowhead_to_tridiag_k_locked_one);
    RUN_TEST(test_pick_locked_orthonormal);
    RUN_TEST(test_restart_state_assemble_roundtrip);
    RUN_TEST(test_restart_state_assemble_n_mismatch_rejected);

    /* Day 3: phase execution + outer loop + backend dispatch. */
    RUN_TEST(test_thick_restart_iterate_empty_state_matches_lanczos);
    RUN_TEST(test_thick_restart_backend_converges_small);
    RUN_TEST(test_thick_restart_backend_smallest);
    RUN_TEST(test_thick_restart_matches_grow_m);
    RUN_TEST(test_thick_restart_backend_eigenvectors);
    RUN_TEST(test_thick_restart_backend_bad_enum_rejected);

    /* Day 4: memory-bounded convergence, parity, AUTO dispatch. */
    RUN_TEST(test_thick_restart_bcsstk14_bounded_memory);
    RUN_TEST(test_thick_restart_parity_nos4);
    RUN_TEST(test_thick_restart_parity_bcsstk04);
    RUN_TEST(test_thick_restart_parity_bcsstk14);
    RUN_TEST(test_thick_restart_auto_dispatch_above_threshold);
    RUN_TEST(test_thick_restart_auto_dispatch_below_threshold);

    TEST_SUITE_END();
}
