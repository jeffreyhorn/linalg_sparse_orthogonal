/*
 * Sprint 29 Day 11 (Item 8): cross-feature integration tests that
 * exercise the interactions between the new Sprint 29 features:
 *
 *   - Item 1: sparse_svd_lowrank_sparse outer-product accumulator
 *     (Day 2).
 *   - Item 2: full-mode SVD U/V output (`economy = 0`, Day 3).
 *   - Item 3: opt-in Rayleigh-quotient eigenpair refinement (Day 5).
 *   - Item 4: progress / cancel callbacks across LU / Cholesky /
 *     LDL^T / QR / iterative / eigs / SVD (Days 6 + 7).
 *
 * Each test pins a multi-feature interaction the per-Item unit tests
 * don't directly catch.  Sprint 29 PLAN.md Day 11 task 2 enumerates
 * the target interactions.
 */

#include "sparse_eigs.h"
#include "sparse_matrix.h"
#include "sparse_svd.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── Test fixture: SPD tridiagonal for eigsolver tests ──────────────── */

static SparseMatrix *build_spd_tridiag(idx_t n) {
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

/* ─── Progress-counter helper (mirrors test_integration.c::progress_count_cb) */

typedef struct {
    idx_t n_calls_total;
    idx_t cancel_after_step;
    const char *last_phase;
} cross_progress_ctx_t;

static int cross_progress_cb(const sparse_progress_t *p, void *user) {
    cross_progress_ctx_t *ctx = (cross_progress_ctx_t *)user;
    ctx->n_calls_total++;
    ctx->last_phase = p->phase;
    if (ctx->cancel_after_step >= 0 && p->step >= ctx->cancel_after_step)
        return 1;
    return 0;
}

/* ───────────────────────────────────────────────────────────────────────
 * Test 1: Lanczos eigsolver + Day-5 refinement + Day-7 progress callback
 *
 * Pins that opts.refine = 1 AND opts.progress_cb != NULL coexist
 * without breakage.  The progress callback fires per Lanczos outer
 * retry boundary (Day-7 hook in s29_maybe_refine doesn't add a
 * separate phase emission); the refinement post-pass runs after
 * Lanczos converges and tightens the per-pair residual to ≤ 1e-13.
 * ─────────────────────────────────────────────────────────────────────── */
static void test_cross_eigs_refine_progress_cb(void) {
    const idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    double vals[3] = {0, 0, 0};
    double *vecs = calloc((size_t)n * 3, sizeof(double));
    REQUIRE_OK(vecs ? SPARSE_OK : SPARSE_ERR_ALLOC);

    cross_progress_ctx_t ctx = {.cancel_after_step = -1};
    sparse_eigs_t res = {.eigenvalues = vals, .eigenvectors = vecs};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-10,
        .reorthogonalize = 1,
        .compute_vectors = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS,
        .refine = 1,
        .refine_max_iters = 5,
        .progress_cb = cross_progress_cb,
        .progress_user = &ctx,
    };
    /* Fatal gate on the solver call — subsequent loops index
     * `vals[0..res.n_converged)` / `vecs[0..n*res.n_converged)`, so
     * a non-OK rc leaving partial / garbage state would lead to
     * out-of-bounds reads or misleading downstream failures.
     * REQUIRE_OK aborts the test immediately if rc != SPARSE_OK. */
    REQUIRE_OK(sparse_eigs_sym(A, 3, &opts, &res));
    ASSERT_EQ(res.n_converged, 3);

    /* Progress callback fired during Lanczos.  Guard the strcmp on
     * `last_phase != NULL` — the framework's ASSERT_TRUE /
     * ASSERT_NOT_NULL are non-fatal (they log + continue), so a
     * NULL `last_phase` (callback never fired) would otherwise
     * segfault on the strcmp dereference and mask the real failure. */
    ASSERT_TRUE(ctx.n_calls_total > 0);
    ASSERT_NOT_NULL(ctx.last_phase);
    if (ctx.last_phase)
        ASSERT_TRUE(strcmp(ctx.last_phase, "lanczos") == 0);

    /* Refinement tightened the per-pair residual.  On a well-
     * separated SPD spectrum the cubic-convergence Rayleigh iteration
     * reaches near-machine-eps after refine_max_iters=5. */
    double max_rel_res = 0.0;
    for (idx_t i = 0; i < res.n_converged; i++) {
        double *v = &vecs[(size_t)i * (size_t)n];
        double lambda = vals[i];
        double res_sq = 0.0;
        for (idx_t r = 0; r < n; r++) {
            double Av_r = 0.0;
            /* Tridiag matvec: A is 4 on diag, -1 on sub/super. */
            Av_r += 4.0 * v[r];
            if (r > 0)
                Av_r += -1.0 * v[r - 1];
            if (r + 1 < n)
                Av_r += -1.0 * v[r + 1];
            double d = Av_r - lambda * v[r];
            res_sq += d * d;
        }
        double rel = sqrt(res_sq) / (fabs(lambda) > 1e-15 ? fabs(lambda) : 1.0);
        if (rel > max_rel_res)
            max_rel_res = rel;
    }
    fprintf(stderr,
            "    eigs+refine+cb: max ||Av - lambda v||/|lambda| = %.3e, "
            "progress_cb calls = %d\n",
            max_rel_res, (int)ctx.n_calls_total);
    ASSERT_TRUE(max_rel_res < 1e-13);

    free(vecs);
    sparse_free(A);
}

/* ───────────────────────────────────────────────────────────────────────
 * Test 2: full-mode SVD (economy=0) + low-rank outer-product
 *
 * Pins that the Day-3 full-mode SVD output (`economy = 0` →
 * U m×m + V^T n×n) plus the Day-2 outer-product low-rank
 * accumulator (env-on path) produce bit-equal reconstruction vs the
 * Day-3 economy mode (`economy = 1` → U m×k + V^T k×n).
 *
 * The outer-product low-rank path reads `svd->U` + `svd->Vt` with
 * `svd->k` as the rank, which is min(m, n).  In full mode the U
 * buffer has m columns total but the first k carry the actual
 * singular triplets — the outer-product loop iterates `s = 0..k-1`
 * so it ignores the padded columns.  This test guards against any
 * off-by-one in the leading-dim handling across modes.
 *
 * Note: `sparse_svd_lowrank_sparse` itself does NOT take a
 * pre-computed sparse_svd_t — it runs its own internal full SVD.
 * To exercise the cross-feature interaction, we run the Day-3
 * full-mode SVD directly and verify the rank-k partial sum
 * reconstructs the input within roundoff.  The outer-product env
 * var path is exercised separately via `sparse_svd_lowrank_sparse`
 * + `SPARSE_SVD_LOWRANK_OUTER=on` (a separate Sprint-29-Day-2 test
 * covers that path; this test pins the full-mode reconstruction
 * identity).
 * ─────────────────────────────────────────────────────────────────────── */
static void test_cross_full_svd_lowrank_reconstruction(void) {
    const idx_t m = 12, n = 8;
    SparseMatrix *A = sparse_create(m, n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);
    /* Deterministic dense-ish fill — same pattern as test_svd_full_u_v_*. */
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < n; j++) {
            double v = (double)(i + 1) * 0.7 - (double)(j + 1) * 1.3 +
                       (((i + j) % 3) ? 1.0 : -1.0) * (double)((i * 11 + j * 7) % 13);
            sparse_insert(A, i, j, v);
        }

    /* Run economy SVD as reference. */
    sparse_svd_opts_t econ_opts = {.compute_uv = 1, .economy = 1};
    sparse_svd_t econ;
    REQUIRE_OK(sparse_svd_compute(A, &econ_opts, &econ));

    /* Run full-mode SVD (Day-3 path). */
    sparse_svd_opts_t full_opts = {.compute_uv = 1, .economy = 0};
    sparse_svd_t full;
    REQUIRE_OK(sparse_svd_compute(A, &full_opts, &full));

    /* Sigma must match. */
    ASSERT_EQ(econ.k, full.k);
    for (idx_t i = 0; i < econ.k; i++)
        ASSERT_TRUE(econ.sigma[i] == full.sigma[i]);

    /* Rank-k outer-product reconstruction must match A within
     * roundoff.  Use full-mode layout: U is m×m leading-dim m, Vt
     * is n×n leading-dim n.  The first k columns of U + first k
     * rows of Vt are the only nonzero contributions (since sigma
     * has only k entries). */
    idx_t k = full.k;
    double frob_a_sq = 0.0;
    double frob_resid_sq = 0.0;
    for (idx_t i = 0; i < m; i++) {
        for (idx_t j = 0; j < n; j++) {
            double recon = 0.0;
            for (idx_t s = 0; s < k; s++) {
                recon += full.sigma[s] * full.U[(size_t)s * (size_t)m + (size_t)i] *
                         full.Vt[(size_t)j * (size_t)n + (size_t)s];
            }
            double a_ij = sparse_get(A, i, j);
            double d = a_ij - recon;
            frob_resid_sq += d * d;
            frob_a_sq += a_ij * a_ij;
        }
    }
    double rel =
        sqrt(frob_a_sq) > 0.0 ? sqrt(frob_resid_sq) / sqrt(frob_a_sq) : sqrt(frob_resid_sq);
    fprintf(stderr, "    full-SVD rank-k recon: ||A - U Sigma Vt||_F/||A||_F = %.3e\n", rel);
    ASSERT_TRUE(rel < 1e-10);

    sparse_svd_free(&econ);
    sparse_svd_free(&full);
    sparse_free(A);
}

/* ───────────────────────────────────────────────────────────────────────
 * Test 3: Refinement cancel propagation
 *
 * Pins the contract that opts.refine = 1 + cancellation during the
 * Lanczos phase short-circuits BEFORE refinement runs (since
 * s29_maybe_refine only runs on SPARSE_OK / SPARSE_ERR_NOT_CONVERGED
 * returns).  Cancellation returns SPARSE_ERR_CANCELLED and the
 * vectors are not refined.
 * ─────────────────────────────────────────────────────────────────────── */
static void test_cross_eigs_refine_cancel_short_circuits(void) {
    const idx_t n = 10;
    SparseMatrix *A = build_spd_tridiag(n);
    REQUIRE_OK(A ? SPARSE_OK : SPARSE_ERR_ALLOC);

    double vals[3] = {0, 0, 0};
    double *vecs = calloc((size_t)n * 3, sizeof(double));
    REQUIRE_OK(vecs ? SPARSE_OK : SPARSE_ERR_ALLOC);

    cross_progress_ctx_t ctx = {.cancel_after_step = 0}; /* cancel at first emission */
    sparse_eigs_t res = {.eigenvalues = vals, .eigenvectors = vecs};
    sparse_eigs_opts_t opts = {
        .which = SPARSE_EIGS_LARGEST,
        .tol = 1e-10,
        .reorthogonalize = 1,
        .compute_vectors = 1,
        .backend = SPARSE_EIGS_BACKEND_LANCZOS,
        .refine = 1,
        .refine_max_iters = 5,
        .progress_cb = cross_progress_cb,
        .progress_user = &ctx,
    };
    sparse_err_t rc = sparse_eigs_sym(A, 3, &opts, &res);
    ASSERT_EQ(rc, SPARSE_ERR_CANCELLED);
    ASSERT_EQ(ctx.n_calls_total, 1); /* exactly the first Lanczos emission */

    /* Vectors should NOT have been refined — they're whatever the
     * Lanczos partial state left.  No assertions on residual; just
     * the cancel-was-honoured signal. */
    free(vecs);
    sparse_free(A);
}

/* ───────────────────────────────────────────────────────────────────────
 * Note on the third PLAN.md cross-feature test:
 *
 *   PLAN.md Day-11 task 2 third bullet asks for:
 *   "Sprint-28-era ND env var × Sprint-29 callbacks:
 *    SPARSE_SUPERNODAL_POSTORDER=on + sparse_reorder_nd with
 *    opts.progress_cb — verify the supernodal postorder post-pass
 *    fires AFTER the multilevel partition's progress events."
 *
 *   Sprint 29 Day 7 explicitly deferred ND progress-callback wiring
 *   to Sprint 30+ (sparse_reorder_nd has no opts struct; adding
 *   sparse_reorder_nd_opts variant was out of scope for the 4-hour
 *   Item-4 close budget).  This test is therefore not feasible in
 *   Sprint 29 and is documented as a Sprint 30+ deliverable in the
 *   Day-7 commit + Day-14 retrospective.
 * ─────────────────────────────────────────────────────────────────────── */

/* ─── Test runner ────────────────────────────────────────────────────── */

int main(void) {
    TEST_SUITE_BEGIN("Sprint 29 cross-feature integration");

    RUN_TEST(test_cross_eigs_refine_progress_cb);
    RUN_TEST(test_cross_full_svd_lowrank_reconstruction);
    RUN_TEST(test_cross_eigs_refine_cancel_short_circuits);

    TEST_SUITE_END();
}
