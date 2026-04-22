/**
 * @file sparse_eigs.c
 * @brief Sparse symmetric eigensolvers — Sprint 20 Days 7-12.
 *
 * Day 7 lands the public API in `include/sparse_eigs.h` plus this
 * compile-ready stub of `sparse_eigs_sym()`.  The stub validates
 * inputs and returns `SPARSE_ERR_BADARG` ("stub in progress"
 * signal; this codebase has no SPARSE_ERR_NOT_IMPL) so Days 8-11
 * have a target to replace without header churn.
 *
 * The implementation roadmap per SPRINT_20/PLAN.md:
 *
 *   Day 8  — basic 3-term Lanczos recurrence on A: build V (Lanczos
 *            vectors) and T (tridiagonal) without reorthogonalization.
 *            Unit-tested on diagonal / tridiagonal fixtures where T
 *            spectrum matches A.
 *   Day 9  — full reorthogonalization pass gated on
 *            `opts->reorthogonalize`: subtract projection onto every
 *            prior Lanczos vector (MGS) to keep V^T V ≈ I under
 *            finite precision.  Wide-spectrum test confirms
 *            ||V^T V - I||_max <= 1e-10.
 *   Day 10 — thick-restart mechanism (Wu/Simon, Stathopoulos/Saad
 *            2007): after m iterations, pick the top k Ritz pairs,
 *            replace V with V·Y_k, resume iteration from the
 *            trailing-beta residual.
 *   Day 11 — Ritz extraction, convergence bookkeeping, and the
 *            full public-API shape (n_converged, iterations,
 *            residual_norm; partial-result SPARSE_ERR_NOT_CONVERGED
 *            on max_iterations exhaustion).
 *   Day 12 — shift-invert mode for SPARSE_EIGS_NEAREST_SIGMA:
 *            factor (A - sigma*I) via sparse_ldlt_factor_opts
 *            (Sprint 20 Day 5 CSC dispatch), apply its inverse at
 *            every Lanczos step, post-process Ritz values via
 *            lambda = sigma + 1/theta.
 */

#include "sparse_eigs.h"

#include "sparse_eigs_internal.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"

#include <math.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Lanczos — 3-term recurrence (Sprint 20 Day 8)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * `lanczos_iterate_basic` builds an m-step Lanczos basis V and
 * tridiagonal T from a symmetric matrix A and starting vector v0.
 * Day 8 implements the recurrence without reorthogonalization; Day
 * 9 layers full MGS reorth on top.
 *
 * The classical 3-term recurrence:
 *
 *     v_0 = v0 / ‖v0‖
 *     for k = 0, 1, 2, ..., m-1:
 *         w       = A · v_k - beta_{k-1} · v_{k-1}   (beta_{-1} := 0)
 *         alpha_k = <w, v_k>
 *         w       = w - alpha_k · v_k
 *         beta_k  = ‖w‖
 *         if beta_k ≈ 0:  invariant subspace — stop
 *         v_{k+1} = w / beta_k
 *
 * Tridiagonal representation: alpha[0..m-1] on T's main diagonal,
 * beta[0..m-2] on the sub/super-diagonal (T is symmetric).
 *
 * Theory (Paige 1972, Parlett 1980).  Let K_m(A, v0) be the Krylov
 * subspace span(v0, A·v0, ..., A^{m-1}·v0).  In exact arithmetic V's
 * columns form an orthonormal basis of K_m and T = V^T·A·V is the
 * projection of A onto that subspace.  T's eigenvalues — the Ritz
 * values — approximate A's eigenvalues, and the approximation is
 * sharpest at the extremes of the spectrum.  That's why Day 11 will
 * pick `LARGEST` / `SMALLEST` Ritz values from T once the
 * thick-restart mechanism keeps m bounded.
 *
 * Finite-precision caveat.  Without reorthogonalization V^T·V
 * drifts from I as k grows and "ghost" Ritz values — duplicates of
 * eigenvalues that already converged — appear in T's spectrum.
 * Paige's analysis (1972) shows ghosts arrive around k ≈ condition-
 * number-scale steps.  Day 9's full reorth suppresses this; Day 8's
 * basic recurrence is sufficient for the unit tests below where m
 * ≤ n on well-conditioned small fixtures.
 *
 * Early-exit rule.  When `beta_k < 1e-14` the recurrence has hit
 * an A-invariant subspace: v_{k+1} would be `w/0` undefined, and
 * T's spectrum up to step k is already a subset of A's spectrum
 * (exact Ritz values, not approximations).  The helper returns
 * SPARSE_OK with *m_actual = k + 1.
 *
 * Day 8 scope.  This is the basic recurrence only: no reorth, no
 * thick-restart, no Ritz extraction.  The full `sparse_eigs_sym`
 * body lands in Day 11; Day 8 exercises the recurrence through
 * unit tests that invoke it directly via sparse_eigs_internal.h
 * (diagonal-spectrum test + tridiagonal identity test). */

sparse_err_t lanczos_iterate_basic(const SparseMatrix *A, const double *v0, idx_t m_max, double *V,
                                   double *alpha, double *beta, idx_t *m_actual) {
    if (!A || !v0 || !V || !alpha || !beta || !m_actual)
        return SPARSE_ERR_NULL;
    idx_t n = sparse_rows(A);
    if (n != sparse_cols(A))
        return SPARSE_ERR_SHAPE;
    if (m_max < 1 || m_max > n)
        return SPARSE_ERR_BADARG;

    *m_actual = 0;

    /* Normalize v0 into V[:, 0]. */
    double v0_sqnorm = 0.0;
    for (idx_t i = 0; i < n; i++)
        v0_sqnorm += v0[i] * v0[i];
    double v0_norm = sqrt(v0_sqnorm);
    if (v0_norm < 1e-14) {
        /* Degenerate starting vector — no direction to iterate in.
         * Matches the spirit of an invariant-subspace exit (the
         * Krylov subspace has dimension 0). */
        return SPARSE_ERR_BADARG;
    }
    {
        double inv = 1.0 / v0_norm;
        for (idx_t i = 0; i < n; i++)
            V[i + 0 * n] = v0[i] * inv;
    }

    /* Scratch for w = A·v_k - beta_{k-1}·v_{k-1} - alpha_k·v_k. */
    double *w = malloc((size_t)n * sizeof(double));
    if (!w)
        return SPARSE_ERR_ALLOC;

    double beta_prev = 0.0; /* beta_{k-1}; zero on the first step */

    for (idx_t k = 0; k < m_max; k++) {
        const double *v_k = V + k * n;

        /* w = A · v_k */
        sparse_matvec(A, v_k, w);

        /* w -= beta_{k-1} · v_{k-1}  (zero contribution on k == 0) */
        if (k > 0) {
            const double *v_prev = V + (k - 1) * n;
            for (idx_t i = 0; i < n; i++)
                w[i] -= beta_prev * v_prev[i];
        }

        /* alpha_k = <w, v_k> */
        double a = 0.0;
        for (idx_t i = 0; i < n; i++)
            a += w[i] * v_k[i];
        alpha[k] = a;

        /* w -= alpha_k · v_k */
        for (idx_t i = 0; i < n; i++)
            w[i] -= a * v_k[i];

        /* beta_k = ‖w‖ */
        double b_sq = 0.0;
        for (idx_t i = 0; i < n; i++)
            b_sq += w[i] * w[i];
        double b = sqrt(b_sq);
        beta[k] = b;

        /* Invariant-subspace detection: w has become the zero
         * vector, so span(V[:, 0..k]) is A-invariant and the
         * Krylov basis has maximal dimension.  Stop cleanly. */
        if (b < 1e-14) {
            *m_actual = k + 1;
            free(w);
            return SPARSE_OK;
        }

        /* Normalise w into v_{k+1} when there's room.  On the
         * final step (k == m_max - 1) we've already filled beta
         * but skip the next-vector write because V has no slot
         * for it. */
        if (k + 1 < m_max) {
            double inv = 1.0 / b;
            double *v_next = V + (k + 1) * n;
            for (idx_t i = 0; i < n; i++)
                v_next[i] = w[i] * inv;
        }
        beta_prev = b;
    }

    *m_actual = m_max;
    free(w);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public entry — `sparse_eigs_sym`
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_eigs_sym(const SparseMatrix *A, idx_t k, const sparse_eigs_opts_t *opts,
                             sparse_eigs_t *result) {
    /* Input validation — shared by the Day 7 stub and the full
     * implementation Days 8-11 will land.  Keep these checks
     * tight so the stub rejects malformed calls (preconditions
     * the public doxygen documents) before returning the Day 7
     * "stub in progress" signal. */
    if (!A || !result)
        return SPARSE_ERR_NULL;
    idx_t n = sparse_rows(A);
    if (n != sparse_cols(A))
        return SPARSE_ERR_SHAPE;
    if (k < 1 || k > n)
        return SPARSE_ERR_BADARG;

    /* Library defaults when opts == NULL match the doxygen
     * contract in sparse_eigs.h. */
    const sparse_eigs_opts_t defaults = {
        .which = SPARSE_EIGS_LARGEST,
        .sigma = 0.0,
        .max_iterations = 0,
        .tol = 0.0,
        .reorthogonalize = 1,
        .compute_vectors = 0,
        .backend = SPARSE_EIGS_BACKEND_AUTO,
    };
    const sparse_eigs_opts_t *o = opts ? opts : &defaults;

    if (o->which != SPARSE_EIGS_LARGEST && o->which != SPARSE_EIGS_SMALLEST &&
        o->which != SPARSE_EIGS_NEAREST_SIGMA)
        return SPARSE_ERR_BADARG;
    if (o->backend != SPARSE_EIGS_BACKEND_AUTO && o->backend != SPARSE_EIGS_BACKEND_LANCZOS)
        return SPARSE_ERR_BADARG;
    if (o->tol < 0.0 || o->max_iterations < 0)
        return SPARSE_ERR_BADARG;
    if (!result->eigenvalues)
        return SPARSE_ERR_NULL;
    if (o->compute_vectors && !result->eigenvectors)
        return SPARSE_ERR_NULL;

    /* Initialise the output fields so partial consumers see
     * deterministic values even on the stub error path.
     * `n_requested` is always set because it is purely a self-
     * describing echo of `k`. */
    result->n_requested = k;
    result->n_converged = 0;
    result->iterations = 0;
    result->residual_norm = 0.0;

    /* Symmetry check is deferred to Day 8 when the iteration body
     * actually needs it — leaving it to the stub now would force
     * building a symmetric fixture just to see SPARSE_ERR_BADARG,
     * which doesn't add coverage over the Day 7 plumbing tests. */

    /* Day 7 stub: input validation above is complete; Days 8-11
     * replace the body with the thick-restart Lanczos
     * implementation described in the module design block above.
     * Until then every call past the validation gates returns
     * SPARSE_ERR_BADARG. */
    return SPARSE_ERR_BADARG;
}
