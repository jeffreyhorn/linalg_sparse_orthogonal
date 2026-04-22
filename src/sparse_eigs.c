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

#include "sparse_dense.h"
#include "sparse_eigs_internal.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"

#include <math.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Lanczos — 3-term recurrence + optional reorthogonalization
 *           (Sprint 20 Days 8-9)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * `lanczos_iterate` builds an m-step Lanczos basis V and
 * tridiagonal T from a symmetric matrix A and starting vector v0.
 * Day 8 implemented the recurrence without reorthogonalization;
 * Day 9 added the `reorthogonalize` gate that layers full MGS
 * reorth on top.
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
 * Reorthogonalization (Day 9).  When the caller sets
 * `reorthogonalize != 0`, after the standard 3-term recurrence
 * produces the tentative w (A·v_k minus the beta_{k-1}·v_{k-1}
 * and alpha_k·v_k pieces), the helper subtracts the projection
 * of w onto every stored Lanczos vector V[:, 0..k):
 *
 *     for j = 0, 1, ..., k-1:
 *         dot  = <w, v_j>
 *         w   -= dot · v_j
 *
 * This is modified Gram-Schmidt (MGS) — numerically more stable
 * than classical Gram-Schmidt at the same asymptotic cost because
 * each subtraction uses the current partially-orthogonalized w
 * rather than a cached dot-product of the original w.  Under MGS
 * the orthogonality drift scales with O(eps · cond(V[:, 0..k))),
 * which for the Krylov bases we build stays at 1e-12 or better up
 * to moderate k.  Classical Gram-Schmidt at comparable cost can
 * lose orthogonality down to 1e-6 or worse on wide-spectrum A.
 *
 * A "twice-MGS" refinement (two passes of the inner j-loop)
 * recovers orthogonality to machine precision on pathological
 * inputs at 2× the reorth cost.  Not currently wired — if Day 11
 * convergence tests show lingering orthogonality drift, add a
 * `opts->reorthogonalize == 2` escalation.
 *
 * Day 9 scope.  Basic recurrence + optional full MGS reorth.  No
 * thick-restart (Day 10), no Ritz extraction (Day 11).  Unit
 * tests via sparse_eigs_internal.h exercise both paths through
 * lanczos_iterate directly. */

sparse_err_t lanczos_iterate(const SparseMatrix *A, const double *v0, idx_t m_max,
                             int reorthogonalize, double *V, double *alpha, double *beta,
                             idx_t *m_actual) {
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

        /* Full MGS reorthogonalization against V[:, 0..k).  The
         * loop iterates j = 0..k-1 so v_k itself is skipped —
         * it's already orthogonal to w after the alpha_k
         * subtraction above.  Each projection uses the current
         * partially-orthogonalized w; that's what distinguishes
         * MGS from classical Gram-Schmidt.  Skipped entirely
         * when `reorthogonalize == 0` (Day 8 baseline). */
        if (reorthogonalize && k > 0) {
            for (idx_t j = 0; j < k; j++) {
                const double *v_j = V + j * n;
                double dot = 0.0;
                for (idx_t i = 0; i < n; i++)
                    dot += w[i] * v_j[i];
                for (idx_t i = 0; i < n; i++)
                    w[i] -= dot * v_j[i];
            }
        }

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
 * Thick-restart Lanczos outer loop (Sprint 20 Day 10)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Day 10 wires `sparse_eigs_sym` into an outer-loop driver that:
 *
 *   1. Starts from a deterministic pseudo-random v0 (golden-ratio
 *      fractional mixing — avoids alignment with any eigenvector
 *      of diagonal test fixtures, reproducible across runs).
 *   2. Runs `lanczos_iterate(A, v0, m, reorth=1, ...)` twice — once
 *      at m = m_short (default 2k + 20) and once at m = m_long
 *      (default m_short + k + 10) — both from the same v0.
 *   3. Extracts Ritz values θ[0..m-1] (sorted ascending) from each
 *      run's T via `tridiag_qr_eigenvalues` (destructively on
 *      local copies of alpha / beta).
 *   4. Stability-based convergence check: compare the top-k θ
 *      values from the two runs.  If they agree to
 *      eff_tol · |θ_j| for every j, the k extreme Ritz values
 *      have converged — their insensitivity to the choice of m
 *      is itself the convergence signal (no Y-based Wu/Simon
 *      residual needed, which is the approach Day 11 can add
 *      later when full eigenvector computation lands).
 *   5. Selects the `k` values matching `opts->which` (LARGEST
 *      from the top of the ascending θ array; SMALLEST from the
 *      bottom) and emits.  NEAREST_SIGMA is deferred to Day 12.
 *   6. On non-convergence, restarts: use the last Lanczos vector
 *      of the longer run (best approximation of the extreme Ritz
 *      direction) as the next v0.  Extends the iteration budget
 *      and retries.  Simpler than Wu/Simon's full arrowhead-state
 *      thick-restart; Day 11 can upgrade if convergence tests
 *      show it's insufficient.
 *
 * Scope deferred to Day 11/12:
 *   - Eigenvector output (`opts->compute_vectors`): needs Y from
 *     the tridiagonal eigenproblem, which `tridiag_qr_eigenvalues`
 *     doesn't produce.  Returns SPARSE_ERR_BADARG when requested
 *     on Day 10.
 *   - Shift-invert mode (`SPARSE_EIGS_NEAREST_SIGMA`): needs the
 *     Day 4-6 LDL^T factor of `A - sigma·I`.  Returns
 *     SPARSE_ERR_BADARG on Day 10.
 *   - Wu/Simon per-pair residual estimators (requires Y).  Day
 *     11 adds them; Day 10 uses the cheaper stability proxy. */

/* Deterministic starting vector — golden-ratio fractional mixing.
 * Avoids alignment with any standard-basis eigenvector (diagonal
 * fixtures would otherwise terminate Lanczos in one step) and is
 * reproducible across runs. */
static void s20_lanczos_starting_vector(double *v0, idx_t n) {
    for (idx_t i = 0; i < n; i++) {
        double x = (double)(i + 1) * 0.618033988749895;
        v0[i] = 0.3 + (x - floor(x));
    }
}

/* Quick approximation of ||A||_inf via scaling the final beta
 * against the largest Ritz value.  If theta_max dominates the
 * iteration's spectrum, ||A||_inf is at least |theta_max|; we
 * use that as the tolerance anchor. */
static double s20_spectrum_scale(const double *theta, idx_t m) {
    double s = 0.0;
    for (idx_t i = 0; i < m; i++) {
        double a = fabs(theta[i]);
        if (a > s)
            s = a;
    }
    return s;
}

/* Compute ascending Ritz values of T from (alpha, beta).
 * Destructive on local copies of alpha / beta inside the helper;
 * caller's alpha / beta are preserved.  On success,
 * theta_out[0..m-1] is sorted ascending. */
static sparse_err_t s20_ritz_values(const double *alpha, const double *beta, idx_t m,
                                    double *theta_out, double *subdiag_scratch) {
    for (idx_t i = 0; i < m; i++)
        theta_out[i] = alpha[i];
    for (idx_t i = 0; i + 1 < m; i++)
        subdiag_scratch[i] = beta[i];
    if (m >= 2)
        return tridiag_qr_eigenvalues(theta_out, subdiag_scratch, m, 0);
    return SPARSE_OK;
}

sparse_err_t sparse_eigs_sym(const SparseMatrix *A, idx_t k, const sparse_eigs_opts_t *opts,
                             sparse_eigs_t *result) {
    /* Input validation (preconditions from the public doxygen). */
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

    result->n_requested = k;
    result->n_converged = 0;
    result->iterations = 0;
    result->residual_norm = 0.0;

    /* Day 10 scope: eigenvector output + shift-invert deferred. */
    if (o->compute_vectors)
        return SPARSE_ERR_BADARG;
    if (o->which == SPARSE_EIGS_NEAREST_SIGMA)
        return SPARSE_ERR_BADARG;

    /* Effective tolerance and iteration budget.  The library
     * defaults match sparse_eigs.h: tol = 1e-10, max_iterations =
     * max(10*k + 20, 100). */
    double eff_tol = o->tol > 0.0 ? o->tol : 1e-10;
    idx_t max_iters = o->max_iterations > 0 ? o->max_iterations : (10 * k + 20);
    if (max_iters < 100)
        max_iters = 100;

    /* Per-batch Lanczos sizes.  m_short is the initial batch; m_long
     * is the stability-check batch (slightly larger so θ top-k can
     * stabilise between the two). */
    idx_t m_short = 2 * k + 20;
    if (m_short > n)
        m_short = n;
    if (m_short < 1)
        m_short = 1;
    idx_t m_long = m_short + k + 10;
    if (m_long > n)
        m_long = n;
    idx_t m_max_alloc = m_long;

    /* Allocate workspace shared across restarts: Lanczos basis V,
     * tridiagonal factors, scratch for tridiag_qr_eigenvalues.
     * Use calloc for scalars the static analyzer can't prove
     * get filled before use (theta_long consumed by the
     * partial-result fallback; v0 consumed by lanczos_iterate). */
    double *V = malloc((size_t)n * (size_t)m_max_alloc * sizeof(double));
    double *alpha = malloc((size_t)m_max_alloc * sizeof(double));
    double *beta = malloc((size_t)m_max_alloc * sizeof(double));
    double *v0 = calloc((size_t)n, sizeof(double));
    double *theta_short = malloc((size_t)m_max_alloc * sizeof(double));
    double *theta_long = calloc((size_t)m_max_alloc, sizeof(double));
    double *subdiag = malloc((size_t)m_max_alloc * sizeof(double));
    if (!V || !alpha || !beta || !v0 || !theta_short || !theta_long || !subdiag) {
        free(V);
        free(alpha);
        free(beta);
        free(v0);
        free(theta_short);
        free(theta_long);
        free(subdiag);
        return SPARSE_ERR_ALLOC;
    }

    s20_lanczos_starting_vector(v0, n);

    idx_t total_iters = 0;
    sparse_err_t rc = SPARSE_ERR_NOT_CONVERGED;
    double last_max_res = 0.0;
    idx_t last_m_actual = 0;

    for (idx_t restart = 0;; restart++) {
        /* Short batch. */
        idx_t budget = max_iters - total_iters;
        idx_t m_this_short = m_short < budget ? m_short : budget;
        if (m_this_short < 2)
            break;
        idx_t m_actual_s = 0;
        sparse_err_t err = lanczos_iterate(A, v0, m_this_short, 1, V, alpha, beta, &m_actual_s);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
        total_iters += m_actual_s;
        err = s20_ritz_values(alpha, beta, m_actual_s, theta_short, subdiag);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }

        /* Long batch (same v0, fresh Lanczos run). */
        budget = max_iters - total_iters;
        idx_t m_this_long = m_long < budget ? m_long : budget;
        if (m_this_long < m_this_short + 1)
            m_this_long = m_this_short + 1;
        if (m_this_long > n)
            m_this_long = n;
        if (m_this_long < 2) {
            /* Out of budget before we could run the stability
             * check.  Emit partial results from the short batch. */
            last_m_actual = m_actual_s;
            break;
        }

        idx_t m_actual_l = 0;
        err = lanczos_iterate(A, v0, m_this_long, 1, V, alpha, beta, &m_actual_l);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
        total_iters += m_actual_l;
        err = s20_ritz_values(alpha, beta, m_actual_l, theta_long, subdiag);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }

        /* Stability check: compare top-k θ from the two batches.
         * `theta_short` is sorted ascending of length m_actual_s;
         * `theta_long` similarly of length m_actual_l.  For LARGEST,
         * compare the last k entries of each; for SMALLEST, the
         * first k.  A pair is "converged" when
         *     |θ_long[j] − θ_short[j]| ≤ eff_tol · max(|θ_long[j]|, scale).
         * All k pairs must agree for the call to converge. */
        idx_t take_s = k < m_actual_s ? k : m_actual_s;
        idx_t take_l = k < m_actual_l ? k : m_actual_l;
        idx_t take = take_s < take_l ? take_s : take_l;
        /* Defensive clamps (satisfy static analyzers that don't
         * deduce take <= min(m_actual_s, m_actual_l) from the
         * construction above).  By construction these are no-ops. */
        if (take > m_actual_s)
            take = m_actual_s;
        if (take > m_actual_l)
            take = m_actual_l;
        if (take < 1)
            break;

        double scale = s20_spectrum_scale(theta_long, m_actual_l);
        int converged = 1;
        double max_diff = 0.0;
        for (idx_t j = 0; j < take; j++) {
            /* Invariants enforced above: 1 <= take <= min(m_actual_s,
             * m_actual_l).  The analyzer can't always deduce this,
             * so guard explicitly. */
            if (j >= m_actual_s || j >= m_actual_l)
                break;
            idx_t idx_s = (o->which == SPARSE_EIGS_LARGEST) ? (m_actual_s - 1 - j) : j;
            idx_t idx_l = (o->which == SPARSE_EIGS_LARGEST) ? (m_actual_l - 1 - j) : j;
            if (idx_s < 0 || idx_s >= m_actual_s)
                break;
            if (idx_l < 0 || idx_l >= m_actual_l)
                break;
            double tv_s = theta_short[idx_s];
            double tv_l = theta_long[idx_l];
            double diff = fabs(tv_l - tv_s);
            double anchor = fabs(tv_l);
            if (anchor < scale * 1e-12)
                anchor = scale > 0.0 ? scale : 1.0;
            if (diff > eff_tol * anchor)
                converged = 0;
            if (diff > max_diff)
                max_diff = diff;
        }

        /* Also converge if Lanczos terminated with an invariant
         * subspace (m_actual_l < m_this_long means beta_k ≈ 0 for
         * some k — T is block-reduced and its Ritz values in that
         * block are exact). */
        int invariant = (m_actual_l < m_this_long);

        if (converged || invariant) {
            /* Emit top-k θ from the longer (more accurate) run. */
            for (idx_t j = 0; j < take; j++) {
                if (j >= m_actual_l)
                    break;
                idx_t idx = (o->which == SPARSE_EIGS_LARGEST) ? (m_actual_l - 1 - j) : j;
                if (idx < 0 || idx >= m_actual_l)
                    break;
                result->eigenvalues[j] = theta_long[idx];
            }
            result->n_converged = take;
            result->iterations = total_iters;
            result->residual_norm = max_diff;
            rc = (take == k) ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
            goto cleanup;
        }

        /* Not converged — save current state for the
         * partial-result fallback and restart. */
        last_max_res = max_diff;
        last_m_actual = m_actual_l;

        /* Restart: use the last long-run Lanczos vector as v0.
         * Guard against m_actual_l == 0 (degenerate case where
         * Lanczos would have returned an error; defensive here). */
        if (m_actual_l < 1)
            break;
        const double *v_last = V + (size_t)(m_actual_l - 1) * (size_t)n;
        for (idx_t i = 0; i < n; i++)
            v0[i] = v_last[i];

        if (total_iters >= max_iters)
            break;

        /* Defensive: avoid pathological infinite loops.  The
         * max_iters budget normally dominates, but if m_batch is
         * tiny this cap prevents unbounded restarts. */
        if (restart > max_iters / m_short + 10)
            break;
    }

    /* Reached here only via the budget-exhausted / restart-limit
     * break.  Emit the best Ritz values from the most recent long
     * batch as a partial NOT_CONVERGED result. */
    if (last_m_actual > 0) {
        idx_t take = k < last_m_actual ? k : last_m_actual;
        for (idx_t j = 0; j < take; j++) {
            if (j >= last_m_actual)
                break;
            idx_t idx = (o->which == SPARSE_EIGS_LARGEST) ? (last_m_actual - 1 - j) : j;
            if (idx < 0 || idx >= last_m_actual)
                break;
            result->eigenvalues[j] = theta_long[idx];
        }
        result->iterations = total_iters;
        result->residual_norm = last_max_res;
    }

cleanup:
    free(V);
    free(alpha);
    free(beta);
    free(v0);
    free(theta_short);
    free(theta_long);
    free(subdiag);
    return rc;
}
