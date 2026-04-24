/**
 * @file sparse_eigs.c
 * @brief Sparse symmetric eigensolvers — Sprint 20 Days 7-12 +
 *        Sprint 21 thick-restart Lanczos.
 *
 * Sprint 20 shipped the symmetric Lanczos eigensolver end-to-end:
 * public API (Day 7), 3-term recurrence (Day 8), full MGS
 * reorthogonalization (Day 9), outer-loop restart mechanism
 * (Day 10, superseded by the Day 13 Wu/Simon redesign), Ritz
 * extraction + convergence gate (Day 11), shift-invert mode for
 * SPARSE_EIGS_NEAREST_SIGMA (Day 12), and the grow-m outer-loop
 * redesign that actually converges on SuiteSparse (Day 13).
 *
 * The Sprint 20 grow-m outer loop holds the full Lanczos basis V
 * across retries — peak memory `O(m_cap · n)`, which on bcsstk14
 * (n = 1806) reaches ~26 MB at m_cap = n.  Sprint 21 Day 1 begins
 * the thick-restart replacement that bounds peak memory at
 * `O((k_locked + m_restart) · n)` by preserving only the
 * converged Ritz subspace across restarts (Wu/Simon 2000;
 * Stathopoulos/Saad 2007).
 *
 * The Sprint 21 roadmap per SPRINT_21/PLAN.md:
 *
 *   Day 1  — arrowhead restart state (`lanczos_restart_state_t`)
 *            + `lanczos_thick_restart_iterate` signature + design
 *            block.  Compile-ready stub returning
 *            SPARSE_ERR_BADARG.
 *   Day 2  — arrowhead-to-tridiagonal Givens reduction +
 *            Ritz-locking helper that forms V_locked = V · Y_k
 *            and packs the restart state.
 *   Day 3  — phase execution: chain Lanczos phases through a
 *            restart state; outer loop composing iterate +
 *            restart + convergence.  Opt-in backend dispatch via
 *            `SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`.
 *   Day 4  — memory-bounded convergence tests + cross-backend
 *            parity vs grow-m on the Sprint 20 corpus.
 *   Days 5-6 — OpenMP reorth parallelism (both backends).
 *   Days 7-10 — LOBPCG solver + AUTO dispatch.
 *   Day 11 — permanent `benchmarks/bench_eigs.c` driver.
 *   Days 12-14 — tests, docs, retrospective.
 */

#include "sparse_eigs.h"

#include "sparse_dense.h"
#include "sparse_eigs_internal.h"
#include "sparse_ldlt.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Portable overflow-safe multiplication: returns 0 on success, 1 on overflow.
 * Mirrors the helper in sparse_qr.c / sparse_svd.c. */
static int size_mul_overflow(size_t a, size_t b, size_t *result) {
    if (a != 0 && b > SIZE_MAX / a)
        return 1;
    *result = a * b;
    return 0;
}

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
 * Early-exit rule.  When `beta_k` falls below the implementation's
 * breakdown tolerance — a scale-aware threshold based on the running
 * `t_norm` (running max of row-k row-sums of T), with a
 * `DBL_MIN * 100` floor for the exact-zero-operator case — the
 * recurrence has hit an A-invariant subspace: v_{k+1} would be
 * `w / beta_k` numerically unstable, and T's spectrum up to step k
 * is already a subset of A's spectrum (exact Ritz values, not
 * approximations).  The helper returns SPARSE_OK with
 * *m_actual = k + 1.
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

/* Default matvec operator: `y = A · x` via `sparse_matvec`.  Used
 * by `lanczos_iterate` as the thin wrapper over
 * `lanczos_iterate_op`. */
static sparse_err_t s20_op_matvec(const void *ctx, idx_t n, const double *x, double *y) {
    (void)n;
    return sparse_matvec((const SparseMatrix *)ctx, x, y);
}

sparse_err_t lanczos_iterate(const SparseMatrix *A, const double *v0, idx_t m_max,
                             int reorthogonalize, double *V, double *alpha, double *beta,
                             idx_t *m_actual) {
    if (!A)
        return SPARSE_ERR_NULL;
    idx_t n = sparse_rows(A);
    if (n != sparse_cols(A))
        return SPARSE_ERR_SHAPE;
    return lanczos_iterate_op(s20_op_matvec, A, n, v0, m_max, reorthogonalize, V, alpha, beta,
                              m_actual);
}

sparse_err_t lanczos_iterate_op(lanczos_op_fn op, const void *ctx, idx_t n, const double *v0,
                                idx_t m_max, int reorthogonalize, double *V, double *alpha,
                                double *beta, idx_t *m_actual) {
    if (!op || !v0 || !V || !alpha || !beta || !m_actual)
        return SPARSE_ERR_NULL;
    if (n < 1)
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

    /* Scratch for w = op·v_k - beta_{k-1}·v_{k-1} - alpha_k·v_k.
     * Overflow-check `n * sizeof(double)` so a pathological n on a
     * 32-bit size_t target fails cleanly rather than undersizing w
     * and corrupting memory in the recurrence loop below. */
    size_t w_bytes = 0;
    if (size_mul_overflow((size_t)n, sizeof(double), &w_bytes))
        return SPARSE_ERR_ALLOC;
    double *w = malloc(w_bytes);
    if (!w)
        return SPARSE_ERR_ALLOC;

    double beta_prev = 0.0; /* beta_{k-1}; zero on the first step */
    /* Running estimate of ||T||_inf used to scale the invariant-
     * subspace / breakdown check.  After each step we update this to
     * the max row-sum |beta_{k-1}| + |alpha_k| + |beta_k| seen so
     * far; beta_k is considered an invariant-subspace trip only
     * when it has dropped well below that accumulated scale.  A
     * purely absolute threshold would falsely fire on small-norm
     * operators (e.g., ||A||_inf ~ 1e-16) where beta_k remains
     * large relative to T but small in absolute terms. */
    double t_norm = 0.0;

    for (idx_t k = 0; k < m_max; k++) {
        const double *v_k = V + k * n;

        /* w = op · v_k — either sparse_matvec(A) for the default
         * path or (A - sigma*I)^{-1} via LDL^T solve for shift-
         * invert mode (Day 12). */
        sparse_err_t op_rc = op(ctx, n, v_k, w);
        if (op_rc != SPARSE_OK) {
            free(w);
            return op_rc;
        }

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

        /* Update the running ||T||_inf estimate with row k's
         * completed row-sum: T[k, k-1] = beta_prev, T[k, k] =
         * alpha_k, T[k, k+1] = b (symmetric tridiagonal). */
        double row_k_bound = beta_prev + fabs(a) + b;
        if (row_k_bound > t_norm)
            t_norm = row_k_bound;

        /* Invariant-subspace detection: w has become the zero
         * vector (or close enough), so span(V[:, 0..k]) is op-
         * invariant and the Krylov basis has maximal dimension.
         * The threshold is scale-aware — `t_norm * 1e-14` handles
         * normal and small-norm operators proportionally, and the
         * `DBL_MIN * 100` absolute floor still triggers on the
         * zero-operator case where `t_norm` stays exactly 0. */
        double breakdown_tol = t_norm * 1e-14;
        if (breakdown_tol < DBL_MIN * 100.0)
            breakdown_tol = DBL_MIN * 100.0;
        if (b < breakdown_tol) {
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

/* Day 11: Ritz pair extraction — returns eigenvalues of the
 * tridiagonal T in `theta_out` (ascending) and T's orthonormal
 * eigenvectors as columns of `Y_out` (m × m, column-major) so the
 * Day 11 lift V · Y[:, j] produces full-problem Ritz vectors.
 * Preserves the caller's alpha / beta; the subdiag_scratch buffer
 * is written to then destroyed by `tridiag_qr_eigenpairs`. */
static sparse_err_t s20_ritz_pairs(const double *alpha, const double *beta, idx_t m,
                                   double *theta_out, double *Y_out, double *subdiag_scratch) {
    for (idx_t i = 0; i < m; i++)
        theta_out[i] = alpha[i];
    for (idx_t i = 0; i + 1 < m; i++)
        subdiag_scratch[i] = beta[i];
    return tridiag_qr_eigenpairs(theta_out, subdiag_scratch, Y_out, m, 0);
}

/* Day 12: shift-invert Lanczos operator.  `ctx` is a
 * `(const sparse_ldlt_t *)` pointing at a pre-computed LDL^T
 * factorisation of `A - sigma*I`.  Applying the operator to a vector
 * `x` means solving `(A - sigma*I) y = x`, i.e. `y = (A - sigma*I)^{-1}
 * x` — exactly the transform that makes Lanczos converge on
 * interior eigenvalues of A (whichever λ_j is closest to σ becomes
 * the largest-magnitude eigenvalue of the shift-inverted operator).
 * Any downstream `sparse_ldlt_solve` error (SPARSE_ERR_SINGULAR,
 * SPARSE_ERR_BADARG) propagates up through `lanczos_iterate_op`. */
static sparse_err_t s20_op_shift_invert(const void *ctx, idx_t n, const double *x, double *y) {
    (void)n;
    return sparse_ldlt_solve((const sparse_ldlt_t *)ctx, x, y);
}

/* Select `min(k_want, m)` indices into `theta[0..m)` by `which`:
 *
 *   LARGEST       - descending theta (sel_idx[0] = m - 1 etc.)
 *   SMALLEST      - ascending theta (sel_idx[0] = 0 etc.)
 *   NEAREST_SIGMA - descending |theta| via a two-pointer sweep over
 *                   the ascending list (largest-|theta| lives at one
 *                   of the two ends; under shift-invert this means
 *                   the Ritz value closest to σ in the original
 *                   lambda-space).
 *
 * Assumes theta is sorted ascending (as `tridiag_qr_eigenpairs`
 * returns it).  Returns the number of indices written. */
static idx_t s20_select_indices(const double *theta, idx_t m, sparse_eigs_which_t which,
                                idx_t k_want, idx_t *sel_idx) {
    idx_t take = k_want < m ? k_want : m;
    if (take < 1)
        return 0;
    if (which == SPARSE_EIGS_LARGEST) {
        for (idx_t j = 0; j < take; j++)
            sel_idx[j] = m - 1 - j;
    } else if (which == SPARSE_EIGS_SMALLEST) {
        for (idx_t j = 0; j < take; j++)
            sel_idx[j] = j;
    } else {
        /* NEAREST_SIGMA: largest-|theta| first.  Two-pointer scan;
         * left runs up from 0, right runs down from m-1.  The loop
         * body bounds-checks both pointers so a partial overlap at
         * the centre of the array can't under/overflow. */
        idx_t left = 0;
        idx_t right = m - 1;
        for (idx_t j = 0; j < take; j++) {
            if (left > right)
                break;
            if (fabs(theta[left]) > fabs(theta[right])) {
                sel_idx[j] = left;
                left++;
            } else {
                sel_idx[j] = right;
                if (right == 0)
                    break;
                right--;
            }
        }
    }
    return take;
}

/* Ritz vector lift: for each j in [0, take), write column j of
 * `eigenvectors_out` (n × take, column-major) with
 *   eigenvector_j = V · Y[:, idx_j]
 * where V is the Lanczos basis (n × m, column-major) and idx_j is
 * the m-space column index of the j-th selected Ritz pair.  Assumes
 * V's columns are already orthonormal (guaranteed by Day 9's full
 * reorth) so the lifted vectors inherit unit norm up to the MGS
 * drift bound (‖ε‖ ≲ 1e-12 on well-conditioned A).  Ritz vectors of
 * (A - σI)^{-1} are also eigenvectors of A (same eigenspaces), so
 * the same lift works for shift-invert mode. */
static void s20_lift_ritz_vectors(const double *V, const double *Y, idx_t n, idx_t m, idx_t take,
                                  const idx_t *idx, double *eigenvectors_out) {
    for (idx_t j = 0; j < take; j++) {
        const double *y = Y + (size_t)idx[j] * (size_t)m;
        double *out = eigenvectors_out + (size_t)j * (size_t)n;
        for (idx_t i = 0; i < n; i++)
            out[i] = 0.0;
        for (idx_t c = 0; c < m; c++) {
            double yc = y[c];
            if (yc == 0.0)
                continue;
            const double *v_c = V + (size_t)c * (size_t)n;
            for (idx_t i = 0; i < n; i++)
                out[i] += yc * v_c[i];
        }
    }
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

    /* Day 11: enforce symmetry precondition.  Matches the Cholesky /
     * IC convention (both call sparse_is_symmetric(A, 1e-12) and
     * return SPARSE_ERR_NOT_SPD on false).  Lanczos silently produces
     * complex or nonsensical Ritz values on a non-symmetric operator,
     * so rejecting at entry is the defensive choice. */
    if (!sparse_is_symmetric(A, 1e-12))
        return SPARSE_ERR_NOT_SPD;

    result->n_requested = k;
    result->n_converged = 0;
    result->iterations = 0;
    result->residual_norm = 0.0;
    result->used_csc_path_ldlt = 0;

    /* Day 12: shift-invert setup for NEAREST_SIGMA.  Factor
     * `A - sigma*I` via `sparse_ldlt_factor_opts` (Days 4-6 AUTO
     * dispatch — benefits from CSC supernodal on big symmetric
     * indefinite shifts) and swap the Lanczos operator from
     * `sparse_matvec(A)` to `sparse_ldlt_solve(ldlt)`.  The
     * factorisation is owned by this call; freed in `cleanup:`.
     * If `A - sigma*I` is singular (σ is exactly an eigenvalue of A),
     * LDL^T reports `SPARSE_ERR_SINGULAR` and we propagate — the
     * public doxygen tells callers to perturb σ slightly in that
     * case. */
    SparseMatrix *A_shifted = NULL;
    sparse_ldlt_t ldlt_shift = {0}; /* zeroed so sparse_ldlt_free is safe */
    lanczos_op_fn op_fn = s20_op_matvec;
    const void *op_ctx = A;
    if (o->which == SPARSE_EIGS_NEAREST_SIGMA) {
        A_shifted = sparse_copy(A);
        if (!A_shifted)
            return SPARSE_ERR_ALLOC;
        for (idx_t i = 0; i < n; i++) {
            double dii = sparse_get(A_shifted, i, i);
            sparse_err_t err = sparse_set(A_shifted, i, i, dii - o->sigma);
            if (err != SPARSE_OK) {
                sparse_free(A_shifted);
                return err;
            }
        }
        int used_csc_path = 0;
        sparse_ldlt_opts_t ldlt_opts = {
            .reorder = SPARSE_REORDER_NONE,
            .tol = 0.0,
            .backend = SPARSE_LDLT_BACKEND_AUTO,
            .used_csc_path = &used_csc_path,
        };
        sparse_err_t err = sparse_ldlt_factor_opts(A_shifted, &ldlt_opts, &ldlt_shift);
        result->used_csc_path_ldlt = used_csc_path;
        if (err != SPARSE_OK) {
            sparse_ldlt_free(&ldlt_shift);
            sparse_free(A_shifted);
            return err;
        }
        op_fn = s20_op_shift_invert;
        op_ctx = &ldlt_shift;
    }

    /* Effective tolerance and iteration budget.  The library
     * defaults match sparse_eigs.h: tol = 1e-10, max_iterations =
     * max(10*k + 20, 100).  Compute the default in int64_t so large
     * k values don't overflow idx_t (int32) before the min-with-n
     * clamp below catches it.
     *
     * When the caller supplies `opts->max_iterations > 0`, honor it
     * as the user's explicit cap rather than silently bumping it up
     * to the library default's 100-iteration floor — that silent
     * promotion contradicted the documented contract ("0 selects the
     * library default ... positive values are honored").  Reject an
     * explicit cap that is too small to run Lanczos safely (< the
     * per-run m_cap_min of 2k+10, clamped to n for small-n inputs)
     * as SPARSE_ERR_BADARG, consistent with the header's
     * "opts values are invalid" SPARSE_ERR_BADARG return. */
    double eff_tol = o->tol > 0.0 ? o->tol : 1e-10;
    idx_t max_iters;
    if (o->max_iterations > 0) {
        int64_t min_required = (int64_t)2 * (int64_t)k + 10;
        if (min_required > (int64_t)n)
            min_required = (int64_t)n;
        if ((int64_t)o->max_iterations < min_required) {
            sparse_ldlt_free(&ldlt_shift);
            sparse_free(A_shifted);
            return SPARSE_ERR_BADARG;
        }
        max_iters = o->max_iterations;
    } else {
        int64_t def_iters = (int64_t)10 * (int64_t)k + 20;
        if (def_iters < 100)
            def_iters = 100;
        if (def_iters > (int64_t)INT32_MAX)
            def_iters = (int64_t)INT32_MAX;
        max_iters = (idx_t)def_iters;
    }

    /* Day 13 outer-loop redesign: single Lanczos batch with a
     * grow-m-on-retry strategy.  The Day 10-11 short/long stability
     * check converged poorly on SuiteSparse matrices because each
     * "restart" threw away the partial basis and resumed from a
     * warm-start v0 — the top-k Ritz values re-approached the same
     * neighbourhood on every pass without tightening.  Day 13
     * replaces that with a single growing Lanczos run: pick an
     * initial m, build the basis, compute Wu/Simon residuals, and
     * (if not converged) grow m and restart from the same v0 (so
     * the first m_prev steps are bit-for-bit identical and the
     * extra m_new − m_prev steps are where the convergence happens).
     * Because Lanczos with full reorth is deterministic under the
     * same v0, each growth round strictly contains the previous
     * one's information. */

    /* Initial and maximum Lanczos basis sizes.
     *   m_cap: upper bound on any single Lanczos run.  Capped at min(n,
     *          max_iters) so the basis never exceeds the user's budget
     *          or the natural Krylov limit.
     *   m_init: starting size per run.  3k + 30 is enough for most
     *           well-separated top-k extractions on small n; bigger
     *           problems grow by m_grow per retry.
     *   m_grow: additive growth per retry — fixed step avoids a
     *           runaway m^3 reorth cost when bumping.
     *
     * All of the `_ * k + _` expressions are evaluated in int64_t
     * and then clamped to `min(..., n)` before narrowing back to
     * idx_t, so large k values can't overflow the int32 idx_t.  The
     * final m_cap / m_init / m_grow are all ≤ n, which fits idx_t. */
    /* n == 1 is a valid symmetric input with k == 1 (trivial
     * eigenpair).  Clamp the lower bound to `min(2, n)` so the
     * 1×1 case doesn't force m_cap > n and trip `lanczos_iterate_op`'s
     * `m_max <= n` precondition. */
    idx_t m_min = (n >= 2) ? 2 : n;
    idx_t m_cap = max_iters < n ? max_iters : n;
    int64_t m_cap_min = (int64_t)2 * (int64_t)k + 10;
    if ((int64_t)m_cap < m_cap_min) {
        m_cap = (m_cap_min > (int64_t)n) ? n : (idx_t)m_cap_min;
    }
    if (m_cap > n)
        m_cap = n;
    if (m_cap < m_min)
        m_cap = m_min;
    int64_t m_init_wide = (int64_t)3 * (int64_t)k + 30;
    if (m_init_wide > (int64_t)m_cap)
        m_init_wide = (int64_t)m_cap;
    if (m_init_wide < (int64_t)m_min)
        m_init_wide = (int64_t)m_min;
    idx_t m_init = (idx_t)m_init_wide;
    int64_t m_grow_wide = (int64_t)k + 20;
    if (m_grow_wide > (int64_t)m_cap)
        m_grow_wide = (int64_t)m_cap;
    idx_t m_grow = (idx_t)m_grow_wide;

    /* Allocate workspace for the upper-bound Lanczos size so the
     * grow-on-retry path never reallocates.  Y_cap is m_cap × m_cap
     * (eigenvectors of T) — quadratic in m_cap but fine for the
     * practical m_cap we land on.  The multi-factor sizes (V, Y_long)
     * are validated with `size_mul_overflow` so a pathological
     * (n, m_cap) pair on a 32-bit size_t target fails cleanly with
     * SPARSE_ERR_ALLOC rather than undersizing a buffer; calloc()
     * handles its own nmemb*size overflow internally. */
    size_t v_elems = 0, v_bytes = 0;
    size_t y_elems = 0;
    size_t sel_idx_bytes = 0;
    if (size_mul_overflow((size_t)n, (size_t)m_cap, &v_elems) ||
        size_mul_overflow(v_elems, sizeof(double), &v_bytes) ||
        size_mul_overflow((size_t)m_cap, (size_t)m_cap, &y_elems) ||
        size_mul_overflow((size_t)k, sizeof(idx_t), &sel_idx_bytes)) {
        sparse_ldlt_free(&ldlt_shift);
        sparse_free(A_shifted);
        return SPARSE_ERR_ALLOC;
    }
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *V = malloc(v_bytes);
    double *alpha = malloc((size_t)m_cap * sizeof(double));
    double *beta = malloc((size_t)m_cap * sizeof(double));
    double *v0 = calloc((size_t)n, sizeof(double));
    double *theta_long = calloc((size_t)m_cap, sizeof(double));
    double *subdiag = malloc((size_t)m_cap * sizeof(double));
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *Y_long = calloc(y_elems, sizeof(double));
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    idx_t *sel_idx = malloc(sel_idx_bytes);
    if (!V || !alpha || !beta || !v0 || !theta_long || !subdiag || !Y_long || !sel_idx) {
        free(V);
        free(alpha);
        free(beta);
        free(v0);
        free(theta_long);
        free(subdiag);
        free(Y_long);
        free(sel_idx);
        sparse_ldlt_free(&ldlt_shift);
        sparse_free(A_shifted);
        return SPARSE_ERR_ALLOC;
    }

    s20_lanczos_starting_vector(v0, n);

    idx_t total_iters = 0;
    sparse_err_t rc = SPARSE_ERR_NOT_CONVERGED;
    idx_t m = m_init;
    idx_t last_m_actual = 0;
    double last_partial_res = 0.0;

    for (;;) {
        idx_t m_actual = 0;
        sparse_err_t err = lanczos_iterate_op(op_fn, op_ctx, n, v0, m, o->reorthogonalize, V, alpha,
                                              beta, &m_actual);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
        /* `iterations` is documented as the total Lanczos work across
         * all grow-m retries (not just the final run), so accumulate
         * each run's m_actual into the counter. */
        total_iters += m_actual;
        last_m_actual = m_actual;
        /* Defensive: lanczos_iterate_op sets m_actual >= 1 on
         * SPARSE_OK (the invariant-subspace early-exit rule sets
         * m_actual = k + 1 with k >= 0, and otherwise m_actual =
         * m_max >= 1).  The guard lets the analyzer see that the
         * `beta[m_actual - 1]` and `(m_actual - 1)` indexings below
         * are in bounds. */
        if (m_actual < 1)
            break;

        /* Lift Ritz pairs (values + Y matrix).  Preserves the
         * caller's alpha / beta (s20_ritz_pairs copies beta into
         * the subdiag scratch before the destructive QR sweep). */
        err = s20_ritz_pairs(alpha, beta, m_actual, theta_long, Y_long, subdiag);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
        /* beta[m_actual - 1] is the Lanczos β_m residual norm (see
         * lanczos_iterate docstring).  The Wu/Simon residual bound
         * for a Ritz pair (θⱼ, y_j) is |β_m · y_{m-1, j}|. */
        double last_beta = beta[m_actual - 1];

        idx_t take = s20_select_indices(theta_long, m_actual, o->which, k, sel_idx);
        if (take < 1)
            break;

        double scale = s20_spectrum_scale(theta_long, m_actual);

        /* Wu/Simon per-pair residuals — the primary convergence
         * gate.  For a Ritz pair (θⱼ, y_j) the true residual
         * ‖(op)·V·y_j − θⱼ·V·y_j‖ equals |β_m · y_{m-1, j}|
         * (Paige 1972; Bai et al. 2000).  max_res_rel is the
         * worst-case relative residual across the k returned pairs
         * — exactly what the result->residual_norm field reports. */
        double max_res_rel = 0.0;
        for (idx_t j = 0; j < take; j++) {
            idx_t idx_l = sel_idx[j];
            double y_last = Y_long[(size_t)(m_actual - 1) + (size_t)idx_l * (size_t)m_actual];
            double abs_res = fabs(last_beta * y_last);
            double tv_l = theta_long[idx_l];
            double anchor = fabs(tv_l);
            if (anchor < scale * 1e-12)
                anchor = scale > 0.0 ? scale : 1.0;
            double rel_res = abs_res / anchor;
            if (rel_res > max_res_rel)
                max_res_rel = rel_res;
        }
        last_partial_res = max_res_rel;

        int converged = (max_res_rel <= eff_tol);
        /* Also converge if Lanczos terminated with an invariant
         * subspace (m_actual < m means β_k ≈ 0 for some k — T is
         * block-reduced and its Ritz values in that block are
         * exact). */
        int invariant = (m_actual < m);

        if (converged || invariant) {
            for (idx_t j = 0; j < take; j++) {
                idx_t idx_l = sel_idx[j];
                double theta = theta_long[idx_l];
                /* For shift-invert (Day 12), the Ritz values of
                 * (A - σI)^{-1} are 1/(λ − σ), so the original-space
                 * eigenvalue is λ = σ + 1/θ.  θ cannot be zero
                 * because (A - σI) was factored nonsingular. */
                result->eigenvalues[j] =
                    (o->which == SPARSE_EIGS_NEAREST_SIGMA) ? (o->sigma + 1.0 / theta) : theta;
            }
            if (o->compute_vectors) {
                s20_lift_ritz_vectors(V, Y_long, n, m_actual, take, sel_idx, result->eigenvectors);
            }
            result->n_converged = take;
            result->iterations = total_iters;
            result->residual_norm = max_res_rel;
            rc = (take == k) ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
            goto cleanup;
        }

        /* Grow m for the next pass.  If we've already hit the cap,
         * emit partial results with NOT_CONVERGED.  The sum
         * `m + m_grow` is computed in int64_t so a runaway
         * combination can't overflow idx_t before the clamp to
         * m_cap caps it back to the valid range.  `max_iterations`
         * controls `m_cap = min(max_iters, n)`, which bounds the
         * Lanczos subspace size per run — not the cumulative work
         * across retries (see field doc in `sparse_eigs.h`). */
        if (m >= m_cap)
            break;
        int64_t m_next_wide = (int64_t)m + (int64_t)m_grow;
        if (m_next_wide > (int64_t)m_cap)
            m_next_wide = (int64_t)m_cap;
        idx_t m_next = (idx_t)m_next_wide;
        if (m_next == m)
            break;
        m = m_next;
    }

    /* Reached here only via m_cap exhaustion without convergence.
     * Emit the best Ritz values (and, if requested, vectors) from
     * the last Lanczos run as a partial NOT_CONVERGED result. */
    if (last_m_actual > 0) {
        idx_t take = s20_select_indices(theta_long, last_m_actual, o->which, k, sel_idx);
        for (idx_t j = 0; j < take; j++) {
            idx_t idx_l = sel_idx[j];
            double theta = theta_long[idx_l];
            result->eigenvalues[j] =
                (o->which == SPARSE_EIGS_NEAREST_SIGMA) ? (o->sigma + 1.0 / theta) : theta;
        }
        if (o->compute_vectors) {
            s20_lift_ritz_vectors(V, Y_long, n, last_m_actual, take, sel_idx, result->eigenvectors);
        }
        result->n_converged = take;
        result->iterations = total_iters;
        result->residual_norm = last_partial_res;
    }

cleanup:
    free(V);
    free(alpha);
    free(beta);
    free(v0);
    free(theta_long);
    free(subdiag);
    free(Y_long);
    free(sel_idx);
    sparse_ldlt_free(&ldlt_shift);
    sparse_free(A_shifted);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 21: Thick-restart Lanczos (Wu/Simon arrowhead)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The Sprint 20 Day 13 grow-m outer loop converges reliably on the
 * target corpus but holds the full Lanczos basis `V` across all
 * retries — peak memory `O(m_cap · n)`.  On bcsstk14 (n = 1806) a
 * single run at `m_cap = n` already reaches ~26 MB of V alone.
 * Sprint 21 Day 1 introduces a thick-restart mechanism (Wu/Simon
 * 2000; Stathopoulos/Saad 2007) that preserves only the converged
 * Ritz subspace across restarts and re-runs Lanczos from a small
 * locked-plus-residual basis.  Peak memory drops to
 * `O((k_locked + m_restart) · n)`; at k = 5, m_restart = 30 this
 * is ~35 columns regardless of total iteration count.
 *
 * Restart protocol (Option C in SPRINT_21/PLAN.md Day 1 Task 3).
 *
 *   1. Run a Lanczos phase of length `m_restart`, writing α / β / V
 *      as usual.  `lanczos_thick_restart_iterate` with an empty
 *      state is equivalent to `lanczos_iterate_op`.
 *   2. On non-convergence, extract Ritz pairs (θ_j, y_j) from T via
 *      the Sprint 20 `s20_ritz_pairs` helper and select the top k
 *      (or matching `which`) via `s20_select_indices` — reused
 *      wholesale.
 *   3. Pack the locked state:
 *        V_locked      = V · Y[:, sel_idx]          (n × k_locked)
 *        theta_locked  = θ[sel_idx]                 (k_locked)
 *        beta_coupling = β_m · Y[m-1, sel_idx]      (k_locked)
 *        residual      = β_m · v_{m+1}              (length n)
 *   4. Launch the next phase with the locked state.  The new phase
 *      copies V_locked into V[:, 0..k_locked), seeds v_{k_locked}
 *      from residual / ||residual||, writes the arrowhead rows
 *      0..k_locked-1 of α / β from theta_locked / beta_coupling,
 *      and continues the 3-term recurrence from step k_locked.
 *   5. Ritz extraction on the arrowhead T → same Ritz pairs the
 *      previous phase produced for the locked block, plus new
 *      approximations from the freshly grown Krylov steps.
 *      Monotone progress is Wu/Simon's core guarantee.
 *
 * Arrowhead T shape (k_locked = 3, m_restart = 7):
 *
 *       [ θ_0   0    0    β_0   0    0    0   ]
 *       [  0   θ_1   0    β_1   0    0    0   ]
 *       [  0    0   θ_2   β_2   0    0    0   ]
 *       [ β_0  β_1  β_2   α_3  β_3   0    0   ]
 *       [  0    0    0    β_3  α_4  β_4   0   ]
 *       [  0    0    0    0    β_4  α_5  β_5  ]
 *       [  0    0    0    0    0    β_5  α_6  ]
 *
 * The top-left k_locked × k_locked block is diagonal (locked Ritz
 * values); the trailing row/column of the block contains the
 * coupling entries β_coupling that tie each locked pair to the
 * active Lanczos frontier; rows k_locked.. are standard tridiagonal.
 * Day 2 implements the Givens chase that reduces this arrowhead
 * back to a symmetric tridiagonal so `tridiag_qr_eigenpairs`
 * (Sprint 20 Day 11) consumes it unchanged.
 *
 * Why keep the grow-m path.  Grow-m is simpler and the constants
 * are good on small-to-moderate n — it converges nos4 (n=100) in
 * 70 Lanczos steps / ~3 ms, bcsstk04 (n=132) in 62 steps / 4.6 ms
 * (Sprint 20 Day 13 numbers in `bench_day13_lanczos.txt`).  The
 * thick-restart path is only a win when the basis would otherwise
 * blow memory.  Day 4 tunes the crossover threshold
 * `SPARSE_EIGS_THICK_RESTART_THRESHOLD`; AUTO dispatches based on
 * n.
 *
 * Field ownership.  `lanczos_restart_state_t` owns its allocations
 * (V_locked / theta_locked / beta_coupling / residual) once
 * populated; `lanczos_restart_state_free` releases them.  An
 * empty state (zeroed struct) is legal input and represents a
 * fresh start.  The Day 2 assembly helpers allocate on first use
 * sized to `k_locked_cap` and reuse the buffers across subsequent
 * restarts when `k_locked <= k_locked_cap` holds (avoids reallocing
 * on every restart — the inner k_locked fluctuates as Ritz pairs
 * lock and the spectrum's active front advances).
 */

void lanczos_restart_state_free(lanczos_restart_state_t *state) {
    if (!state)
        return;
    free(state->V_locked);
    free(state->theta_locked);
    free(state->beta_coupling);
    free(state->residual);
    state->V_locked = NULL;
    state->theta_locked = NULL;
    state->beta_coupling = NULL;
    state->residual = NULL;
    state->n = 0;
    state->k_locked = 0;
    state->k_locked_cap = 0;
    state->residual_norm = 0.0;
}

/* Day 1 stub: V / alpha / beta are output buffers that Day 2-3 fill
 * during the phase-execution body.  The stub doesn't write to them
 * yet, so clang-tidy flags them as const-candidates; the NOLINT
 * suppressions document this as intentional until the body lands. */
sparse_err_t lanczos_thick_restart_iterate(lanczos_op_fn op, const void *ctx, idx_t n,
                                           const double *v0, idx_t m_restart, int reorthogonalize,
                                           lanczos_restart_state_t *state,
                                           double *V,     // NOLINT(readability-non-const-parameter)
                                           double *alpha, // NOLINT(readability-non-const-parameter)
                                           double *beta,  // NOLINT(readability-non-const-parameter)
                                           idx_t *m_actual) {
    /* Day 1 stub.  Validates the core preconditions so Day 2 /
     * Day 3 can replace the body without the public signature
     * drifting.  Returns SPARSE_ERR_BADARG for the success case —
     * the "stub in progress" signal used throughout this codebase
     * (no SPARSE_ERR_NOT_IMPL exists).
     *
     * Day 2 replaces the body with: (a) copy V_locked into V,
     * (b) write arrowhead rows 0..k_locked-1 of α / β from
     * theta_locked / beta_coupling, (c) seed v_{k_locked} from
     * state->residual (orthogonalised against V_locked), then
     * (d) run the standard 3-term recurrence from step k_locked
     * onward.  Day 3 wires this into the outer loop and the
     * public `sparse_eigs_sym` dispatch. */
    if (!op || !V || !alpha || !beta || !m_actual)
        return SPARSE_ERR_NULL;
    if (n < 1)
        return SPARSE_ERR_SHAPE;
    if (m_restart < 1 || m_restart > n)
        return SPARSE_ERR_BADARG;
    /* An empty state (NULL or k_locked == 0) requires v0.  A
     * non-empty state carries its own residual seed so v0 may be
     * NULL. */
    int state_empty = (state == NULL) || (state->k_locked == 0) || (state->V_locked == NULL);
    if (state_empty && !v0)
        return SPARSE_ERR_NULL;
    if (!state_empty) {
        if (state->n != n)
            return SPARSE_ERR_SHAPE;
        if (state->k_locked < 0 || state->k_locked >= m_restart)
            return SPARSE_ERR_BADARG;
        if (state->k_locked > state->k_locked_cap)
            return SPARSE_ERR_BADARG;
        if (!state->theta_locked || !state->beta_coupling || !state->residual)
            return SPARSE_ERR_NULL;
    }
    /* Silence unused-parameter warnings on the fields Day 2 will
     * consume; keeps -Wunused-parameter clean under the stub. */
    (void)ctx;
    (void)reorthogonalize;
    *m_actual = 0;
    return SPARSE_ERR_BADARG;
}
