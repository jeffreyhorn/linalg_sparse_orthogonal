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
 *   Day 7  — LOBPCG API surface + Rayleigh-Ritz infrastructure
 *            stubs (`s21_lobpcg_solve` returns BADARG); design
 *            block landed here, between the Lanczos and
 *            thick-restart blocks.
 *   Days 8-10 — LOBPCG body, preconditioning, AUTO dispatch.
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

#ifdef SPARSE_OPENMP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <omp.h>
#pragma GCC diagnostic pop
#endif

/* ─── Sprint 21 Day 5/6: shared MGS reorth kernel (OpenMP-parallel) ── */

/* Small-n gate for the MGS reorth parallelism.  Below this `n`,
 * per-call OMP fork/join overhead (measured ~5-20 μs on macOS
 * Homebrew libomp) dominates the dot-product + daxpy work, and
 * the parallel reduction is a net loss — the Day 6 scaling sweep
 * showed every thread count > 1 regressed on nos4 (n = 100),
 * bcsstk04 (n = 132), and kkt-150 (n = 150), while bcsstk14
 * (n = 1806) scaled to 2.2× at 4 threads.  500 is a safe
 * crossover: well above the n = 150 loser set, well below the
 * n ≈ 1800 clear winner.  Callers who need a different
 * crossover on their hardware can override by building with
 * `-DSPARSE_EIGS_OMP_REORTH_MIN_N=<value>`.  See
 * SPRINT_21/bench_day6_omp_scaling.txt for the raw sweep and
 * SPRINT_17/PERF_NOTES.md for the threshold rationale. */
#ifndef SPARSE_EIGS_OMP_REORTH_MIN_N
#define SPARSE_EIGS_OMP_REORTH_MIN_N 500
#endif

/* s21_mgs_reorth: orthogonalise `w` against each of V[:, 0..k_stored-1]
 * via classical modified Gram-Schmidt (MGS).  The outer `j` loop
 * is serial — MGS stability requires each iteration to see the
 * partially-orthogonalised `w` from the previous subtraction
 * (classical Gram-Schmidt parallelises `j` but loses the stability
 * bound; see the Sprint 20 Day 9 design block for the MGS-vs-CGS
 * tradeoff).  The inner dot-product and daxpy bodies are
 * independent across `i` and parallelised under `-DSPARSE_OPENMP`,
 * using the same pragma pattern as `sparse_matvec` from
 * Sprint 17/18 (`#pragma omp parallel for`).
 *
 * Day 6 threshold gate: the `if (n >= SPARSE_EIGS_OMP_REORTH_MIN_N)`
 * clause causes each `parallel for` to run serially (a single
 * implicit team of one) when `n` is small enough that OMP
 * overhead would exceed the parallel work.  The clause is a
 * zero-cost no-op in serial builds (the whole pragma is
 * `#ifdef SPARSE_OPENMP`-gated out).  Matvec in Sprint 17/18 has
 * the same overhead structure but does not gate — its tuning
 * lives in a future sprint; Day 6 only addresses MGS reorth.
 *
 * Serial builds are bit-for-bit identical to the Sprint 20 inline
 * MGS body the helper replaced — the Day 5 refactor collapses
 * three call sites (`lanczos_iterate_op`, the thick-restart seed
 * orthogonalisation, the thick-restart phase reorth) into one
 * kernel so parallelism lands once and benefits all three paths.
 *
 * Shared across Sprint 20 (`lanczos_iterate_op`) and Sprint 21
 * (`lanczos_thick_restart_iterate`).  Does nothing when
 * `k_stored == 0`. */
static void s21_mgs_reorth(double *w, const double *V, idx_t n, idx_t k_stored) {
    for (idx_t j = 0; j < k_stored; j++) {
        const double *v_j = V + (size_t)j * (size_t)n;
        double dot = 0.0;
#ifdef SPARSE_OPENMP
#pragma omp parallel for reduction(+ : dot) schedule(static) if (n >= SPARSE_EIGS_OMP_REORTH_MIN_N)
#endif
        for (idx_t i = 0; i < n; i++)
            dot += w[i] * v_j[i];
#ifdef SPARSE_OPENMP
#pragma omp parallel for schedule(static) if (n >= SPARSE_EIGS_OMP_REORTH_MIN_N)
#endif
        for (idx_t i = 0; i < n; i++)
            w[i] -= dot * v_j[i];
    }
}

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
         * when `reorthogonalize == 0` (Day 8 baseline).  Sprint
         * 21 Day 5 shares the kernel with `lanczos_thick_restart_iterate`
         * via the `s21_mgs_reorth` helper, which parallelises the
         * inner dot-product + daxpy under `-DSPARSE_OPENMP`. */
        if (reorthogonalize && k > 0)
            s21_mgs_reorth(w, V, n, k);

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

/* Forward declaration: the Sprint 21 thick-restart outer loop is
 * defined below but dispatched to from `sparse_eigs_sym` here.
 * Keeps the Day 3 dispatch self-contained without moving the
 * thick-restart block above the Sprint 20 code. */
static sparse_err_t s21_thick_restart_outer_loop(lanczos_op_fn op, const void *ctx, idx_t n,
                                                 idx_t k, const sparse_eigs_opts_t *o,
                                                 double eff_tol, idx_t max_iters,
                                                 sparse_eigs_t *result);

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
        .lobpcg_soft_lock = 1, /* Day 9: default-on per the public-header doc. */
    };
    const sparse_eigs_opts_t *o = opts ? opts : &defaults;

    if (o->which != SPARSE_EIGS_LARGEST && o->which != SPARSE_EIGS_SMALLEST &&
        o->which != SPARSE_EIGS_NEAREST_SIGMA)
        return SPARSE_ERR_BADARG;
    if (o->backend != SPARSE_EIGS_BACKEND_AUTO && o->backend != SPARSE_EIGS_BACKEND_LANCZOS &&
        o->backend != SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART &&
        o->backend != SPARSE_EIGS_BACKEND_LOBPCG)
        return SPARSE_ERR_BADARG;
    if (o->tol < 0.0 || o->max_iterations < 0)
        return SPARSE_ERR_BADARG;
    if (!result->eigenvalues)
        return SPARSE_ERR_NULL;
    if (o->compute_vectors && !result->eigenvectors)
        return SPARSE_ERR_NULL;
    /* Sprint 21 Day 7: LOBPCG-specific opts validation.  Block size
     * must be at least k (so the Rayleigh-Ritz step produces enough
     * Ritz pairs); the special value 0 means "library default = k"
     * and is accepted as the designated-init zero default.  The
     * `precond_ctx != NULL && precond == NULL` mismatch is rejected
     * as the obvious user error (forgot to set the callback after
     * binding the context); the inverse `precond != NULL &&
     * precond_ctx == NULL` is allowed because some preconditioners
     * carry state via globals or thread-local storage. */
    if (o->block_size < 0 || o->block_size > n)
        return SPARSE_ERR_BADARG;
    if (o->block_size > 0 && o->block_size < k)
        return SPARSE_ERR_BADARG;
    if (o->precond_ctx && !o->precond)
        return SPARSE_ERR_BADARG;

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
    result->peak_basis_size = 0;
    result->backend_used = SPARSE_EIGS_BACKEND_LANCZOS; /* updated per dispatch below */

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

    /* Sprint 21 Day 3/4: thick-restart backend dispatch.
     *
     * Day 3 opt-in path: `opts->backend ==
     * SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART` forces the
     * Wu/Simon restart pipeline.
     *
     * Day 4 AUTO routing: when `opts->backend == _AUTO` and the
     * problem is large enough (`n >= SPARSE_EIGS_THICK_RESTART_THRESHOLD`,
     * default 500; tuned in `docs/planning/EPIC_2/SPRINT_21/bench_day4_restart.txt`),
     * also route here — the grow-m path's `O(m_cap · n)` peak
     * memory becomes prohibitive on the big SuiteSparse fixtures
     * where the thick-restart's `O((m_restart + k) · n)` bound
     * wins by an order of magnitude or more (bcsstk14 at
     * n = 1806, k = 5: grow-m ~7 MB of V vs thick-restart
     * ~500 KB). */
    /* Sprint 21 Day 7: LOBPCG dispatch.  Day 7 lands the explicit
     * opt-in path only — `opts->backend == SPARSE_EIGS_BACKEND_LOBPCG`
     * routes to `s21_lobpcg_solve` (currently a BADARG stub; Days 8-9
     * fill the body).  Day 10 extends AUTO to also route to LOBPCG
     * above the n / block_size / precond threshold; until then,
     * AUTO continues to pick between Lanczos and thick-restart per
     * the existing size threshold. */
    if (o->backend == SPARSE_EIGS_BACKEND_LOBPCG) {
        result->backend_used = SPARSE_EIGS_BACKEND_LOBPCG;
        sparse_err_t lobpcg_rc =
            s21_lobpcg_solve(op_fn, op_ctx, n, k, o, eff_tol, max_iters, result);
        sparse_ldlt_free(&ldlt_shift);
        sparse_free(A_shifted);
        return lobpcg_rc;
    }

    int use_thick_restart =
        (o->backend == SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART) ||
        (o->backend == SPARSE_EIGS_BACKEND_AUTO && n >= (idx_t)SPARSE_EIGS_THICK_RESTART_THRESHOLD);
    if (use_thick_restart) {
        result->backend_used = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART;
        sparse_err_t tr_rc =
            s21_thick_restart_outer_loop(op_fn, op_ctx, n, k, o, eff_tol, max_iters, result);
        sparse_ldlt_free(&ldlt_shift);
        sparse_free(A_shifted);
        return tr_rc;
    }
    result->backend_used = SPARSE_EIGS_BACKEND_LANCZOS;

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

    /* Day 4 telemetry: grow-m holds V at m_cap columns from
     * allocation until cleanup — peak basis size is `m_cap`
     * regardless of how many grow-on-retry passes actually run. */
    result->peak_basis_size = m_cap;

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

/* ─── Sprint 21 Day 2: Arrowhead reduction + Ritz locking helpers ─── */

/* s21_arrowhead_to_tridiag: reduce a symmetric arrowhead T to
 * tridiagonal form via dense Householder tridiagonalisation.
 *
 * Arrowhead layout (K = k_locked + m_ext):
 *   T[i, i]                 = theta_locked[i]    for i in [0, k_locked)
 *   T[i, i]                 = alpha_ext[i-k_locked]
 *                                                for i in [k_locked, K)
 *   T[k_locked, j]          = beta_coupling[j]    (spoke)
 *   T[j, k_locked]          = beta_coupling[j]    (symmetric)
 *                                                for j in [0, k_locked)
 *   T[i, i+1] = T[i+1, i]   = beta_ext[i-k_locked]
 *                                                for i in [k_locked, K-1)
 *   all other entries are zero.
 *
 * The implementation builds a dense K×K scratch matrix and applies
 * (K-2) Householder similarity transforms, each zeroing the
 * sub-subdiagonal entries of one column.  Classical algorithm from
 * Golub & Van Loan §8.3.1.  Choice notes:
 *   - Householder over Givens here because the arrowhead pattern
 *     produces fill across the locked block under simple spoke-
 *     zeroing Givens (the bulge-chase sequence is equivalent work
 *     to a full Householder on the dense matrix but much harder to
 *     get correct).  Dense Householder is O(K^3); for K up to a
 *     few hundred the cost is a microsecond-scale fixed overhead
 *     per restart.
 *   - Scratch allocation is owned by this function (caller passes
 *     no workspace).  Size is K*K doubles; overflow-checked.
 *   - Spectrum-only reduction (Day 2 scope).  Day 3 extends the
 *     helper to return an orthogonal Q accumulating the Householder
 *     products so downstream Ritz extraction composes through the
 *     reduction.
 */
sparse_err_t s21_arrowhead_to_tridiag(const double *theta_locked, const double *beta_coupling,
                                      idx_t k_locked, const double *alpha_ext,
                                      const double *beta_ext, idx_t m_ext, double *diag_out,
                                      double *subdiag_out) {
    if (!theta_locked || !diag_out)
        return SPARSE_ERR_NULL;
    if (k_locked >= 1 && !beta_coupling)
        return SPARSE_ERR_NULL;
    if (m_ext >= 1 && !alpha_ext)
        return SPARSE_ERR_NULL;
    if (m_ext >= 2 && !beta_ext)
        return SPARSE_ERR_NULL;
    if (k_locked < 1)
        return SPARSE_ERR_BADARG;

    /* K is the dimension of the (k_locked + m_ext) arrowhead.  We
     * require K >= 1; subdiag_out is needed only when K >= 2. */
    int64_t K_wide = (int64_t)k_locked + (int64_t)m_ext;
    if (K_wide < 1 || K_wide > (int64_t)INT32_MAX)
        return SPARSE_ERR_BADARG;
    idx_t K = (idx_t)K_wide;
    if (K >= 2 && !subdiag_out)
        return SPARSE_ERR_NULL;

    /* Dense K×K scratch.  Overflow-check K*K*sizeof(double). */
    size_t K2 = 0, K2_bytes = 0;
    if (size_mul_overflow((size_t)K, (size_t)K, &K2) ||
        size_mul_overflow(K2, sizeof(double), &K2_bytes))
        return SPARSE_ERR_ALLOC;
    double *T = calloc(K2, sizeof(double));
    if (!T)
        return SPARSE_ERR_ALLOC;

    /* Materialise the arrowhead.  Layout column-major: T[i + j*K]. */
#define T_AT(i, j) T[(size_t)(i) + (size_t)(j) * (size_t)K]

    /* K >= k_locked by construction (K_wide = k_locked + m_ext with
     * k_locked >= 1 already checked), but the analyzer can't prove
     * it — suppress the false-positive heap-bound warning. */
    for (idx_t i = 0; i < k_locked; i++)
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        T_AT(i, i) = theta_locked[i];
    for (idx_t i = 0; i < m_ext; i++)
        T_AT(k_locked + i, k_locked + i) = alpha_ext[i];
    if (m_ext >= 2) {
        for (idx_t i = 0; i + 1 < m_ext; i++) {
            idx_t r = k_locked + i;
            T_AT(r + 1, r) = beta_ext[i];
            T_AT(r, r + 1) = beta_ext[i];
        }
    }
    /* Spoke: row k_locked holds beta_coupling, but only when there
     * is a spoke row at all (m_ext >= 1; otherwise the arrowhead is
     * just the locked diagonal and there is no spoke column).  When
     * m_ext >= 1 we also put beta_coupling[k_locked-1] at
     * (k_locked, k_locked-1) as the standard subdiagonal entry
     * connecting the last locked row to the first extension row. */
    if (m_ext >= 1) {
        for (idx_t j = 0; j < k_locked; j++) {
            T_AT(k_locked, j) = beta_coupling[j];
            T_AT(j, k_locked) = beta_coupling[j];
        }
    }

    /* Householder tridiagonalisation of the symmetric K×K matrix.
     * Per iteration j in [0, K-2):
     *   - Let x = T[j+1..K-1, j] (column j below the diagonal).
     *   - If ||x[1:]|| < eps (already reduced), skip.
     *   - Choose sign(alpha) = sign(x[0]) so v = x + alpha*e_0 has
     *     a numerically large magnitude (avoid cancellation).
     *   - Normalise v so that beta_scale = 2 / (v^T v) defines the
     *     reflector H = I - beta_scale * v * v^T.
     *   - Apply the symmetric similarity T -> H T H (affects only
     *     rows and columns j+1..K-1).  Implementation uses the
     *     two-step form: compute p = T_sub * v, then
     *     q = beta_scale * p - (beta_scale^2 / 2) * (v^T p) * v,
     *     and update T_sub -= v * q^T + q * v^T (rank-2 update).
     *   - Write the reduced subdiagonal entry T[j+1, j] = -alpha
     *     directly; zero the rest of column j below the subdiagonal.
     */
    if (K >= 3) {
        /* Scratch buffers sized to K-1 (worst case for column 0). */
        double *v = calloc((size_t)K, sizeof(double));
        double *p = calloc((size_t)K, sizeof(double));
        double *q = calloc((size_t)K, sizeof(double));
        if (!v || !p || !q) {
            free(v);
            free(p);
            free(q);
            free(T);
            return SPARSE_ERR_ALLOC;
        }

        for (idx_t j = 0; j + 2 < K; j++) {
            idx_t len = K - j - 1; /* length of column vector below diagonal */

            /* Extract x = T[j+1..K-1, j] into v. */
            for (idx_t i = 0; i < len; i++)
                v[i] = T_AT(j + 1 + i, j);

            /* sigma = ||x[1:]||^2 = sum_{i>=1} v[i]^2 */
            double sigma = 0.0;
            for (idx_t i = 1; i < len; i++)
                sigma += v[i] * v[i];
            if (sigma < 1e-300) {
                /* Already reduced (or len == 1) — subdiag[j] = v[0];
                 * rest of column j is already zero. */
                continue;
            }

            /* alpha = sign(v[0]) * sqrt(sigma + v[0]^2).  The sign
             * choice matches the "avoid cancellation" rule:
             * v[0]_new = v[0] + sign(v[0]) * ||x|| has large
             * magnitude. */
            double v0 = v[0];
            double x_norm = sqrt(sigma + v0 * v0);
            double alpha = (v0 >= 0.0) ? x_norm : -x_norm;
            v[0] = v0 + alpha;

            /* beta_scale = 2 / (v^T v) = 2 / (sigma + v[0]_new^2)
             * where v[0]_new = v0 + alpha, so
             *   v[0]_new^2 = v0^2 + 2*v0*alpha + alpha^2
             *             = v0^2 + 2*v0*alpha + (sigma + v0^2)
             *             = 2*v0^2 + 2*v0*alpha + sigma
             * and v^T v = sigma + v[0]_new^2 = 2*sigma + 2*v0^2 + 2*v0*alpha
             *           = 2 * (alpha^2 + v0*alpha)
             *           = 2 * alpha * (alpha + v0)
             *           = 2 * alpha * v[0]_new
             * so beta_scale = 1 / (alpha * v[0]_new).
             *
             * Defensive: if the denominator is numerically zero the
             * reflector is ill-conditioned; bail out of this step. */
            double denom = alpha * v[0];
            if (fabs(denom) < 1e-300)
                continue;
            double beta_scale = 1.0 / denom;

            /* p = T_sub * v (T_sub is the (len × len) submatrix
             * T[j+1..K-1, j+1..K-1]). */
            for (idx_t i = 0; i < len; i++) {
                double pi = 0.0;
                for (idx_t c = 0; c < len; c++)
                    pi += T_AT(j + 1 + i, j + 1 + c) * v[c];
                p[i] = pi;
            }

            /* vtp = v^T p */
            double vtp = 0.0;
            for (idx_t i = 0; i < len; i++)
                vtp += v[i] * p[i];

            /* q = beta_scale * p - (beta_scale^2 / 2) * vtp * v */
            double K_coef = 0.5 * beta_scale * beta_scale * vtp;
            for (idx_t i = 0; i < len; i++)
                q[i] = beta_scale * p[i] - K_coef * v[i];

            /* T_sub -= v * q^T + q * v^T (symmetric rank-2 update). */
            for (idx_t i = 0; i < len; i++) {
                double vi = v[i];
                double qi = q[i];
                for (idx_t c = 0; c < len; c++)
                    T_AT(j + 1 + i, j + 1 + c) -= vi * q[c] + qi * v[c];
            }

            /* Write the reduced subdiagonal explicitly (avoid drift
             * from floating-point cancellation in the rank-2 update). */
            T_AT(j + 1, j) = -alpha;
            T_AT(j, j + 1) = -alpha;
            for (idx_t i = 2; i < len; i++) {
                T_AT(j + 1 + i - 1, j) = 0.0;
                T_AT(j, j + 1 + i - 1) = 0.0;
            }
        }

        free(v);
        free(p);
        free(q);
    }

    /* Extract the tridiagonal form.  The Householder loop stops at
     * K-3 because column K-2 is already reduced (only one
     * sub-diagonal entry) and column K-1 has no sub-diagonal.  Both
     * are already in the correct place in T. */
    for (idx_t i = 0; i < K; i++)
        diag_out[i] = T_AT(i, i);
    if (K >= 2) {
        for (idx_t i = 0; i + 1 < K; i++)
            subdiag_out[i] = T_AT(i + 1, i);
    }
#undef T_AT

    free(T);
    return SPARSE_OK;
}

/* lanczos_restart_pick_locked: Day 2 Task 2 — assemble the three
 * locked-block arrays from a completed Lanczos phase's Ritz pairs.
 *
 *   V_locked[:, j] = V · Y[:, sel_idx[j]]
 *   theta_locked[j] = theta[sel_idx[j]]
 *   beta_coupling[j] = beta_m * Y[m-1, sel_idx[j]]
 *
 * Reuses the `s20_lift_ritz_vectors` kernel shape directly for the
 * V · Y[:, idx] column-major gemm. */
void lanczos_restart_pick_locked(const double *V, idx_t n, idx_t m, const double *Y,
                                 const double *theta, const idx_t *sel_idx, idx_t take,
                                 double beta_m, double *V_locked_out, double *theta_locked_out,
                                 double *beta_coupling_out) {
    s20_lift_ritz_vectors(V, Y, n, m, take, sel_idx, V_locked_out);
    for (idx_t j = 0; j < take; j++) {
        idx_t col = sel_idx[j];
        theta_locked_out[j] = theta[col];
        /* Y is column-major m × m: Y[m-1, col] lives at offset
         * (m - 1) + col * m. */
        double y_last = Y[(size_t)(m - 1) + (size_t)col * (size_t)m];
        beta_coupling_out[j] = beta_m * y_last;
    }
}

/* lanczos_restart_state_assemble: Day 2 Task 3 — pack the picked
 * locked block + residual into `state`, allocating buffers if the
 * current capacity is insufficient.  Reuses existing buffers when
 * `k_locked <= state->k_locked_cap` and `state->n == n`. */
sparse_err_t lanczos_restart_state_assemble(lanczos_restart_state_t *state, idx_t n, idx_t k_locked,
                                            const double *V_locked_src,
                                            const double *theta_locked_src,
                                            const double *beta_coupling_src,
                                            const double *residual_src, double residual_norm) {
    if (!state)
        return SPARSE_ERR_NULL;
    if (n < 1 || k_locked < 0)
        return SPARSE_ERR_BADARG;
    if (k_locked > 0 && (!V_locked_src || !theta_locked_src || !beta_coupling_src))
        return SPARSE_ERR_NULL;
    if (!residual_src)
        return SPARSE_ERR_NULL;

    /* If state is non-empty and n mismatches, the caller is trying
     * to reuse a state across different eigenproblems — reject
     * rather than silently reallocating. */
    if (state->n != 0 && state->n != n)
        return SPARSE_ERR_SHAPE;

    /* Allocate or grow V_locked capacity if needed.  We keep the
     * residual buffer sized to n regardless of k_locked, so it's
     * allocated separately below. */
    if (k_locked > state->k_locked_cap) {
        size_t v_elems = 0, v_bytes = 0;
        if (size_mul_overflow((size_t)n, (size_t)k_locked, &v_elems) ||
            size_mul_overflow(v_elems, sizeof(double), &v_bytes))
            return SPARSE_ERR_ALLOC;
        // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
        double *new_V = malloc(v_bytes);
        double *new_theta = malloc((size_t)k_locked * sizeof(double));
        double *new_beta = malloc((size_t)k_locked * sizeof(double));
        if (!new_V || !new_theta || !new_beta) {
            free(new_V);
            free(new_theta);
            free(new_beta);
            return SPARSE_ERR_ALLOC;
        }
        free(state->V_locked);
        free(state->theta_locked);
        free(state->beta_coupling);
        state->V_locked = new_V;
        state->theta_locked = new_theta;
        state->beta_coupling = new_beta;
        state->k_locked_cap = k_locked;
    }

    /* Residual buffer allocated lazily / on first use.  Same n
     * across restarts, so only allocate once. */
    if (!state->residual) {
        state->residual = malloc((size_t)n * sizeof(double));
        if (!state->residual)
            return SPARSE_ERR_ALLOC;
    }

    /* Copy the locked block + residual into state-owned memory. */
    if (k_locked > 0) {
        memcpy(state->V_locked, V_locked_src, (size_t)n * (size_t)k_locked * sizeof(double));
        memcpy(state->theta_locked, theta_locked_src, (size_t)k_locked * sizeof(double));
        memcpy(state->beta_coupling, beta_coupling_src, (size_t)k_locked * sizeof(double));
    }
    memcpy(state->residual, residual_src, (size_t)n * sizeof(double));

    state->n = n;
    state->k_locked = k_locked;
    state->residual_norm = residual_norm;
    return SPARSE_OK;
}

/* lanczos_thick_restart_iterate: Day 3 implementation.
 *
 * Runs one Lanczos phase of length `m_restart` against the
 * symmetric operator `op`.  Two modes:
 *
 *   Empty state (NULL, or k_locked == 0, or V_locked == NULL): the
 *     body delegates directly to `lanczos_iterate_op` — the
 *     phase behaves exactly like a fresh Lanczos run.  This is the
 *     first-phase path in `s21_thick_restart_outer_loop`.
 *
 *   Non-empty state: the body injects the locked Ritz block at the
 *     head of V / alpha / beta, seeds v_{k_locked} from
 *     `state->residual / state->residual_norm` (re-orthogonalised
 *     against V_locked to kill finite-precision drift), and
 *     continues the 3-term recurrence from step k_locked onward.
 *     The arrowhead T's spokes `beta_coupling[j]` are NOT written
 *     into the flat `alpha / beta` arrays for j < k_locked - 1 —
 *     those rows of the arrowhead are off-tridiagonal and the
 *     caller reads them back from `state->beta_coupling` when
 *     building the arrowhead for Ritz extraction.  The last
 *     spoke `beta_coupling[k_locked-1]` IS written as the standard
 *     subdiagonal entry `beta[k_locked - 1]` because it sits on
 *     the natural tridiagonal line between the locked block and
 *     the first extension row.  Full-MGS reorth handles the
 *     implicit spoke subtraction at step k_locked so no explicit
 *     spoke-correction is needed in the recurrence (each new
 *     Lanczos vector is orthogonalised against ALL previously
 *     stored V columns, including the locked block).
 */
sparse_err_t lanczos_thick_restart_iterate(lanczos_op_fn op, const void *ctx, idx_t n,
                                           const double *v0, idx_t m_restart, int reorthogonalize,
                                           lanczos_restart_state_t *state, double *V, double *alpha,
                                           double *beta, idx_t *m_actual) {
    if (!op || !V || !alpha || !beta || !m_actual)
        return SPARSE_ERR_NULL;
    if (n < 1)
        return SPARSE_ERR_SHAPE;
    if (m_restart < 1 || m_restart > n)
        return SPARSE_ERR_BADARG;
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
        if (state->residual_norm <= 0.0)
            return SPARSE_ERR_BADARG; /* invariant-subspace trip; caller should stop */
    }

    *m_actual = 0;

    /* Empty-state fast path: delegates to the Sprint 20 helper. */
    if (state_empty)
        return lanczos_iterate_op(op, ctx, n, v0, m_restart, reorthogonalize, V, alpha, beta,
                                  m_actual);

    idx_t k_locked = state->k_locked;

    /* Copy locked block into V[:, 0..k_locked-1]. */
    memcpy(V, state->V_locked, (size_t)n * (size_t)k_locked * sizeof(double));
    /* alpha[0..k_locked-1] = theta_locked. */
    memcpy(alpha, state->theta_locked, (size_t)k_locked * sizeof(double));
    /* beta[0..k_locked-2] = 0 (locked block is diagonal in T);
     * beta[k_locked-1] = beta_coupling[k_locked-1] (standard
     * subdiagonal connecting the locked block to the first
     * extension row; the preceding k_locked-1 coupling entries
     * are off-tridiagonal spokes that the outer loop reads from
     * state->beta_coupling). */
    for (idx_t i = 0; i + 1 < k_locked; i++)
        beta[i] = 0.0;
    if (k_locked >= 1)
        beta[k_locked - 1] = state->beta_coupling[k_locked - 1];

    /* Seed v_{k_locked} from the state's residual.  MGS-reorthogonalise
     * against V_locked once (the residual should be orthogonal to
     * V_locked by the Lanczos property in exact arithmetic; the
     * reorth pass cleans up finite-precision drift).  Then
     * normalise. */
    double *v_seed = V + (size_t)k_locked * (size_t)n;
    double inv_rn = 1.0 / state->residual_norm;
    for (idx_t i = 0; i < n; i++)
        v_seed[i] = state->residual[i] * inv_rn;
    if (reorthogonalize) {
        s21_mgs_reorth(v_seed, V, n, k_locked);
        double sq = 0.0;
        for (idx_t i = 0; i < n; i++)
            sq += v_seed[i] * v_seed[i];
        double nrm = sqrt(sq);
        if (nrm < DBL_MIN * 100.0) {
            /* Residual collapsed under reorth — invariant subspace
             * was essentially reached.  Report the locked block only. */
            *m_actual = k_locked;
            return SPARSE_OK;
        }
        double inv = 1.0 / nrm;
        for (idx_t i = 0; i < n; i++)
            v_seed[i] *= inv;
    }

    /* Continue the 3-term recurrence from step k_locked onward.
     * Mirrors the Sprint 20 `lanczos_iterate_op` inner body but
     * with a starting step index of k_locked and an augmented
     * V whose first k_locked columns are the locked block.  Full-
     * MGS reorth against V[:, 0..k) at each step handles the
     * arrowhead-spoke subtraction implicitly. */
    size_t w_bytes = 0;
    if (size_mul_overflow((size_t)n, sizeof(double), &w_bytes))
        return SPARSE_ERR_ALLOC;
    double *w = malloc(w_bytes);
    if (!w)
        return SPARSE_ERR_ALLOC;

    /* beta_prev at step k_locked: from the Lanczos relation after
     * restart, v_{k_locked} was seeded from the prior-phase residual
     * and the coupling to V_locked is carried in the arrowhead spokes
     * (not in beta_prev).  For the 3-term recurrence continuation,
     * beta_prev of step k_locked is *0* because v_{k_locked-1} is
     * part of the locked block and the arrowhead subdiagonal
     * inside the locked block is 0.  The MGS reorth pass picks up
     * the spoke coupling from the locked vectors. */
    double beta_prev = 0.0;
    double t_norm = 0.0;

    for (idx_t k = k_locked; k < m_restart; k++) {
        double *v_k = V + (size_t)k * (size_t)n;

        sparse_err_t op_rc = op(ctx, n, v_k, w);
        if (op_rc != SPARSE_OK) {
            free(w);
            return op_rc;
        }

        /* w -= beta_{k-1} · v_{k-1}  (only for the first new step,
         * this evaluates to zero since beta_prev initialises to 0;
         * for subsequent steps it's the standard tridiagonal
         * continuation). */
        if (k > k_locked) {
            const double *v_prev = V + (size_t)(k - 1) * (size_t)n;
            for (idx_t i = 0; i < n; i++)
                w[i] -= beta_prev * v_prev[i];
        }

        double a = 0.0;
        for (idx_t i = 0; i < n; i++)
            a += w[i] * v_k[i];
        alpha[k] = a;

        for (idx_t i = 0; i < n; i++)
            w[i] -= a * v_k[i];

        /* Full MGS reorth against V[:, 0..k).  This is the step
         * where the arrowhead spoke coupling gets absorbed on the
         * first new step (k == k_locked): w is orthogonalised
         * against V_locked columns, which implicitly subtracts
         * beta_coupling[j] · V_locked[:, j] for j in [0, k_locked). */
        if (reorthogonalize && k > 0)
            s21_mgs_reorth(w, V, n, k);

        double b_sq = 0.0;
        for (idx_t i = 0; i < n; i++)
            b_sq += w[i] * w[i];
        double b = sqrt(b_sq);
        beta[k] = b;

        /* Running ||T||_inf for the scale-aware breakdown check. */
        double row_k_bound = beta_prev + fabs(a) + b;
        if (row_k_bound > t_norm)
            t_norm = row_k_bound;

        double breakdown_tol = t_norm * 1e-14;
        if (breakdown_tol < DBL_MIN * 100.0)
            breakdown_tol = DBL_MIN * 100.0;
        if (b < breakdown_tol) {
            *m_actual = k + 1;
            free(w);
            return SPARSE_OK;
        }

        /* Normalise w into v_{k+1} when there's room. */
        if (k + 1 < m_restart) {
            double inv = 1.0 / b;
            double *v_next = V + (size_t)(k + 1) * (size_t)n;
            for (idx_t i = 0; i < n; i++)
                v_next[i] = w[i] * inv;
        }

        beta_prev = b;
    }

    free(w);
    *m_actual = m_restart;
    return SPARSE_OK;
}

/* ─── Sprint 21 Day 3: Dense symmetric eigensolver (Jacobi) ─────── */

/* Classical Jacobi sweeps on a dense symmetric K × K matrix.
 * Returns ascending eigenvalues in `theta_out[0..K-1]` and the
 * corresponding orthonormal eigenvectors as columns of `Q_out`
 * (K × K, column-major).  Used for arrowhead Ritz extraction
 * (Day 3) because the arrowhead doesn't have the tridiagonal
 * shape `tridiag_qr_eigenpairs` needs; bypasses the Day 2
 * reduction-to-tridiag shim so the outer loop doesn't have to
 * thread a separate Q accumulator through the reduction.
 *
 * Cost: O(K^3) per sweep × O(log K) sweeps.  For K ≤ 100 this is
 * microsecond-scale; acceptable as long as m_restart stays bounded.
 *
 * Input `A_scratch` is destroyed (overwritten with the diagonalised
 * form as a side effect). */
static sparse_err_t s21_dense_sym_jacobi(double *A_scratch, idx_t K, double *theta_out,
                                         double *Q_out) {
    if (!A_scratch || !theta_out || !Q_out)
        return SPARSE_ERR_NULL;
    if (K < 1)
        return SPARSE_ERR_BADARG;

    /* Q := I. */
    for (idx_t j = 0; j < K; j++) {
        for (idx_t i = 0; i < K; i++)
            Q_out[(size_t)i + (size_t)j * (size_t)K] = (i == j) ? 1.0 : 0.0;
    }

    if (K == 1) {
        theta_out[0] = A_scratch[0];
        return SPARSE_OK;
    }

    const idx_t max_sweeps = 100;
    const double tol = 1e-14;

    for (idx_t sweep = 0; sweep < max_sweeps; sweep++) {
        /* off-diagonal Frobenius norm */
        double off = 0.0;
        for (idx_t i = 0; i < K; i++) {
            for (idx_t j = i + 1; j < K; j++) {
                double aij = A_scratch[(size_t)i + (size_t)j * (size_t)K];
                off += aij * aij;
            }
        }
        if (sqrt(off) < tol)
            break;

        for (idx_t p = 0; p < K; p++) {
            for (idx_t q = p + 1; q < K; q++) {
                size_t pq = (size_t)p + (size_t)q * (size_t)K;
                double apq = A_scratch[pq];
                if (fabs(apq) < tol)
                    continue;
                double app = A_scratch[(size_t)p + (size_t)p * (size_t)K];
                double aqq = A_scratch[(size_t)q + (size_t)q * (size_t)K];
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (fabs(theta) > 1e15) {
                    t = 1.0 / (2.0 * theta);
                } else {
                    double sign_t = theta >= 0.0 ? 1.0 : -1.0;
                    t = sign_t / (fabs(theta) + sqrt(theta * theta + 1.0));
                }
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                /* Update rows/cols p, q of A (symmetric). */
                for (idx_t i = 0; i < K; i++) {
                    if (i == p || i == q)
                        continue;
                    double aip = A_scratch[(size_t)i + (size_t)p * (size_t)K];
                    double aiq = A_scratch[(size_t)i + (size_t)q * (size_t)K];
                    double new_ip = c * aip - s * aiq;
                    double new_iq = s * aip + c * aiq;
                    A_scratch[(size_t)i + (size_t)p * (size_t)K] = new_ip;
                    A_scratch[(size_t)p + (size_t)i * (size_t)K] = new_ip;
                    A_scratch[(size_t)i + (size_t)q * (size_t)K] = new_iq;
                    A_scratch[(size_t)q + (size_t)i * (size_t)K] = new_iq;
                }
                A_scratch[(size_t)p + (size_t)p * (size_t)K] =
                    c * c * app - 2.0 * s * c * apq + s * s * aqq;
                A_scratch[(size_t)q + (size_t)q * (size_t)K] =
                    s * s * app + 2.0 * s * c * apq + c * c * aqq;
                A_scratch[(size_t)p + (size_t)q * (size_t)K] = 0.0;
                A_scratch[(size_t)q + (size_t)p * (size_t)K] = 0.0;

                /* Update Q's rows p, q (equivalently cols p, q
                 * since we're building Q s.t. A = Q * diag * Q^T;
                 * each rotation is applied from the right to Q). */
                for (idx_t i = 0; i < K; i++) {
                    size_t ip = (size_t)i + (size_t)p * (size_t)K;
                    size_t iq = (size_t)i + (size_t)q * (size_t)K;
                    double qip = Q_out[ip];
                    double qiq = Q_out[iq];
                    Q_out[ip] = c * qip - s * qiq;
                    Q_out[iq] = s * qip + c * qiq;
                }
            }
        }
    }

    /* Sort eigenvalues ascending; permute Q columns to match. */
    for (idx_t i = 0; i < K; i++)
        theta_out[i] = A_scratch[(size_t)i + (size_t)i * (size_t)K];
    /* Simple selection sort — K is small. */
    for (idx_t i = 0; i < K; i++) {
        idx_t min_idx = i;
        for (idx_t j = i + 1; j < K; j++) {
            if (theta_out[j] < theta_out[min_idx])
                min_idx = j;
        }
        if (min_idx != i) {
            double tmp = theta_out[i];
            theta_out[i] = theta_out[min_idx];
            theta_out[min_idx] = tmp;
            for (idx_t r = 0; r < K; r++) {
                double q_tmp = Q_out[(size_t)r + (size_t)i * (size_t)K];
                Q_out[(size_t)r + (size_t)i * (size_t)K] =
                    Q_out[(size_t)r + (size_t)min_idx * (size_t)K];
                Q_out[(size_t)r + (size_t)min_idx * (size_t)K] = q_tmp;
            }
        }
    }

    return SPARSE_OK;
}

/* ─── Sprint 21 Day 3: Thick-restart outer loop ─────────────────── */

/* Compose the arrowhead T (dense K × K) from the flat alpha / beta
 * arrays plus the state's off-tridiagonal spoke entries.  When
 * `k_locked == 0` (fresh phase), T is pure tridiagonal; when
 * `k_locked > 0` the top-left k_locked × k_locked block is
 * diagonal, row/col k_locked carries the spoke `beta_coupling`
 * (with the last entry already in beta[k_locked-1] as standard
 * subdiag), and rows k_locked.. are standard tridiagonal. */
static void s21_build_dense_arrowhead(const double *alpha, const double *beta,
                                      const double *beta_coupling, idx_t k_locked, idx_t K,
                                      double *T_out) {
    memset(T_out, 0, (size_t)K * (size_t)K * sizeof(double));
    for (idx_t i = 0; i < K; i++)
        T_out[(size_t)i + (size_t)i * (size_t)K] = alpha[i];
    if (K >= 2) {
        for (idx_t i = 0; i + 1 < K; i++) {
            T_out[(size_t)(i + 1) + (size_t)i * (size_t)K] = beta[i];
            T_out[(size_t)i + (size_t)(i + 1) * (size_t)K] = beta[i];
        }
    }
    if (k_locked >= 2) {
        /* Spokes at (k_locked, j) for j in [0, k_locked-1); the
         * last coupling entry beta_coupling[k_locked-1] is already
         * at (k_locked, k_locked-1) via the beta subdiagonal fill
         * above. */
        for (idx_t j = 0; j + 1 < k_locked; j++) {
            T_out[(size_t)k_locked + (size_t)j * (size_t)K] = beta_coupling[j];
            T_out[(size_t)j + (size_t)k_locked * (size_t)K] = beta_coupling[j];
        }
    }
}

/* Recompute the unnormalised Lanczos residual
 *   residual = A v_{m-1} − alpha[m-1] v_{m-1} − beta[m-2] v_{m-2}
 * from the completed V / alpha / beta arrays.  This is the
 * Lanczos "overflow" vector that `lanczos_iterate_op` normally
 * normalises into `v_m` when k+1 < m_max; by recomputing it here
 * we avoid threading a residual output through the iterator
 * signature (one extra matvec per restart).  `||residual||`
 * should equal `beta[m-1]` in exact arithmetic. */
static sparse_err_t s21_recompute_residual(lanczos_op_fn op, const void *ctx, idx_t n,
                                           const double *V, const double *alpha, const double *beta,
                                           idx_t m, double *residual_out) {
    if (m < 1)
        return SPARSE_ERR_BADARG;
    const double *v_last = V + (size_t)(m - 1) * (size_t)n;
    sparse_err_t op_rc = op(ctx, n, v_last, residual_out);
    if (op_rc != SPARSE_OK)
        return op_rc;
    for (idx_t i = 0; i < n; i++)
        residual_out[i] -= alpha[m - 1] * v_last[i];
    if (m >= 2) {
        const double *v_prev = V + (size_t)(m - 2) * (size_t)n;
        for (idx_t i = 0; i < n; i++)
            residual_out[i] -= beta[m - 2] * v_prev[i];
    }
    return SPARSE_OK;
}

/* s21_thick_restart_outer_loop: Wu/Simon thick-restart dispatch.
 * Called from `sparse_eigs_sym` when
 * `opts->backend == SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`
 * (Day 3; Day 4 extends AUTO to route here above a size threshold).
 *
 * Manages the phase-by-phase restart loop with bounded memory:
 * V / alpha / beta are sized to `m_restart` (fixed), not the
 * monotone-growing `m_cap` of the Sprint 20 grow-m path.  Peak
 * memory is `O((m_restart + k_locked_cap) · n)`, independent of
 * total iteration count.
 *
 * Convergence gate mirrors the grow-m path: Wu/Simon per-pair
 * residual `|beta_last · Y_arrow[m_actual - 1, j]|` scaled by
 * `max(|theta_j|, scale)`.  When `beta_last` is the last Lanczos
 * beta of the CURRENT phase (not the prior-phase spoke), the
 * identity `||A V_aug y - θ V_aug y|| = |beta_last · y_last|` holds
 * across the augmented (locked + new) subspace by the same Paige
 * derivation the Sprint 20 grow-m path uses.
 *
 * Arguments are the pre-processed outer-loop inputs from
 * `sparse_eigs_sym` (operator + context + shift-invert state).
 * Result is populated on exit. */
static sparse_err_t s21_thick_restart_outer_loop(lanczos_op_fn op, const void *ctx, idx_t n,
                                                 idx_t k, const sparse_eigs_opts_t *o,
                                                 double eff_tol, idx_t max_iters,
                                                 sparse_eigs_t *result) {
    /* Restart basis size.  Day 4 tuning: `2k + 20` keeps peak
     * `V + V_locked` at ~`m_restart + k = 3k + 20` columns, which
     * for `k = 5` gives 35 columns — roughly 15× smaller than the
     * grow-m path's typical `m_cap = 500` while still leaving
     * enough of a Krylov spectrum per phase to converge extreme
     * Ritz values in < 30 restarts on the Sprint 20 SuiteSparse
     * corpus (see
     * `docs/planning/EPIC_2/SPRINT_21/bench_day4_restart.txt`).
     * Capped at `n` and `max_iters`, and floored so `m_restart >
     * k_locked` (the thick-restart iterator precondition). */
    int64_t m_restart_wide = (int64_t)2 * (int64_t)k + 20;
    if (m_restart_wide > (int64_t)n)
        m_restart_wide = (int64_t)n;
    if (m_restart_wide > (int64_t)max_iters)
        m_restart_wide = (int64_t)max_iters;
    if (m_restart_wide < (int64_t)k + 1)
        m_restart_wide = (int64_t)k + 1;
    if (m_restart_wide > (int64_t)n)
        m_restart_wide = (int64_t)n;
    idx_t m_restart = (idx_t)m_restart_wide;

    /* Day 4 telemetry: peak simultaneous V columns = m_restart
     * (main buffer) + k (locked state across restarts) + k (the
     * transient `V_locked_tmp` during pick_locked, briefly live
     * alongside both V and state->V_locked).  On a grow-m run
     * with m_cap = 500, k = 5 this metric lands at 510, so the
     * bcsstk14 parity test captures the memory savings by
     * comparing peak_basis_size ratios rather than absolute
     * numbers. */
    result->peak_basis_size = m_restart + 2 * k;

    /* Workspace.  All sizes bounded by (m_restart, k) regardless
     * of how many restarts the outer loop eventually runs. */
    size_t v_elems = 0, v_bytes = 0;
    size_t K2 = 0, K2_bytes = 0;
    size_t vk_elems = 0, vk_bytes = 0;
    if (size_mul_overflow((size_t)n, (size_t)m_restart, &v_elems) ||
        size_mul_overflow(v_elems, sizeof(double), &v_bytes) ||
        size_mul_overflow((size_t)m_restart, (size_t)m_restart, &K2) ||
        size_mul_overflow(K2, sizeof(double), &K2_bytes) ||
        size_mul_overflow((size_t)n, (size_t)k, &vk_elems) ||
        size_mul_overflow(vk_elems, sizeof(double), &vk_bytes))
        return SPARSE_ERR_ALLOC;

    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *V = malloc(v_bytes);
    double *alpha = malloc((size_t)m_restart * sizeof(double));
    double *beta = malloc((size_t)m_restart * sizeof(double));
    double *v0 = calloc((size_t)n, sizeof(double));
    double *residual_vec = malloc((size_t)n * sizeof(double));
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *T_arrow = malloc(K2_bytes);
    double *theta_arrow = malloc((size_t)m_restart * sizeof(double));
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *Y_arrow = malloc(K2_bytes);
    idx_t *sel_idx = malloc((size_t)k * sizeof(idx_t));
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *V_locked_tmp = malloc(vk_bytes);
    double *theta_locked_tmp = malloc((size_t)k * sizeof(double));
    double *beta_coupling_tmp = malloc((size_t)k * sizeof(double));
    lanczos_restart_state_t state = {0};

    sparse_err_t rc = SPARSE_ERR_NOT_CONVERGED;

    if (!V || !alpha || !beta || !v0 || !residual_vec || !T_arrow || !theta_arrow || !Y_arrow ||
        !sel_idx || !V_locked_tmp || !theta_locked_tmp || !beta_coupling_tmp) {
        rc = SPARSE_ERR_ALLOC;
        goto cleanup;
    }

    s20_lanczos_starting_vector(v0, n);

    /* Outer restart loop.  Total work cap via `max_iters` — each
     * phase contributes (m_actual - k_locked) new Lanczos steps to
     * the cumulative count. */
    idx_t total_iters = 0;
    idx_t last_m_actual = 0;
    idx_t last_take = 0;
    double last_partial_res = 0.0;

    /* Upper bound on phases: each phase does at least 1 new step,
     * so max_restarts = max_iters is safe.  The scale-aware break
     * conditions below exit earlier in practice. */
    for (idx_t phase = 0; phase < max_iters; phase++) {
        if (total_iters >= max_iters)
            break;

        idx_t m_actual = 0;
        sparse_err_t err = lanczos_thick_restart_iterate(
            op, ctx, n, v0, m_restart, o->reorthogonalize, &state, V, alpha, beta, &m_actual);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
        if (m_actual < 1)
            break;

        /* Accumulate new-iteration count.  The first k_locked
         * columns are the locked block (no new Lanczos work),
         * so only m_actual - k_locked counts toward the budget. */
        total_iters += (m_actual > state.k_locked) ? (m_actual - state.k_locked) : 0;
        last_m_actual = m_actual;

        /* Build the arrowhead T and extract Ritz pairs via
         * dense Jacobi (bypasses the Day 2 reduce-to-tridiag
         * helper because Jacobi produces Y directly in the
         * arrowhead basis — no composition of transforms needed). */
        idx_t K = m_actual;
        s21_build_dense_arrowhead(alpha, beta, state.beta_coupling, state.k_locked, K, T_arrow);
        err = s21_dense_sym_jacobi(T_arrow, K, theta_arrow, Y_arrow);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }

        idx_t take = s20_select_indices(theta_arrow, K, o->which, k, sel_idx);
        last_take = take;
        if (take < 1)
            break;

        /* Wu/Simon residual: |beta_last · y_{K-1, j}| scaled by
         * max(|theta_j|, scale).  beta_last is the LAST Lanczos
         * beta of the current phase (beta[m_actual - 1]); on an
         * invariant-subspace early exit this is the breakdown-
         * threshold scalar, which makes the residual tiny. */
        double beta_last = beta[m_actual - 1];
        double scale = s20_spectrum_scale(theta_arrow, K);
        double max_res_rel = 0.0;
        for (idx_t j = 0; j < take; j++) {
            idx_t idx_l = sel_idx[j];
            double y_last = Y_arrow[(size_t)(K - 1) + (size_t)idx_l * (size_t)K];
            double abs_res = fabs(beta_last * y_last);
            double tv_l = theta_arrow[idx_l];
            double anchor = fabs(tv_l);
            if (anchor < scale * 1e-12)
                anchor = scale > 0.0 ? scale : 1.0;
            double rel_res = abs_res / anchor;
            if (rel_res > max_res_rel)
                max_res_rel = rel_res;
        }
        last_partial_res = max_res_rel;

        int converged = (max_res_rel <= eff_tol);
        int invariant = (m_actual < m_restart);

        if (converged || invariant) {
            for (idx_t j = 0; j < take; j++) {
                idx_t idx_l = sel_idx[j];
                double theta = theta_arrow[idx_l];
                result->eigenvalues[j] =
                    (o->which == SPARSE_EIGS_NEAREST_SIGMA) ? (o->sigma + 1.0 / theta) : theta;
            }
            if (o->compute_vectors) {
                s20_lift_ritz_vectors(V, Y_arrow, n, K, take, sel_idx, result->eigenvectors);
            }
            result->n_converged = take;
            result->iterations = total_iters;
            result->residual_norm = max_res_rel;
            rc = (take == k) ? SPARSE_OK : SPARSE_ERR_NOT_CONVERGED;
            goto cleanup;
        }

        /* Not converged and didn't hit an invariant subspace —
         * assemble the next restart state and loop. */
        idx_t k_lock_next = take; /* lock exactly the target set */
        lanczos_restart_pick_locked(V, n, K, Y_arrow, theta_arrow, sel_idx, k_lock_next, beta_last,
                                    V_locked_tmp, theta_locked_tmp, beta_coupling_tmp);

        /* Recompute the unnormalised residual = beta_last · v_{m+1}
         * from the completed V / alpha / beta.  One extra matvec;
         * keeps `lanczos_thick_restart_iterate`'s signature tight. */
        err = s21_recompute_residual(op, ctx, n, V, alpha, beta, m_actual, residual_vec);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }

        /* If the recomputed residual norm has collapsed (numerical
         * invariant subspace that the breakdown check didn't
         * catch — can happen when finite-precision reorth leaves
         * a tiny residual), emit partial results rather than
         * launching a doomed restart. */
        double res_norm_check = 0.0;
        for (idx_t i = 0; i < n; i++)
            res_norm_check += residual_vec[i] * residual_vec[i];
        res_norm_check = sqrt(res_norm_check);
        if (res_norm_check < DBL_MIN * 100.0)
            break;

        err = lanczos_restart_state_assemble(&state, n, k_lock_next, V_locked_tmp, theta_locked_tmp,
                                             beta_coupling_tmp, residual_vec, res_norm_check);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
    }

    /* Budget or restart cap reached without convergence.  Emit
     * partial results from the last phase (matches Sprint 20
     * grow-m path's final-phase fallthrough). */
    if (last_m_actual > 0 && last_take > 0) {
        /* Re-run the selection + lift from the last phase's
         * already-cached theta_arrow / Y_arrow / sel_idx. */
        for (idx_t j = 0; j < last_take; j++) {
            idx_t idx_l = sel_idx[j];
            double theta = theta_arrow[idx_l];
            result->eigenvalues[j] =
                (o->which == SPARSE_EIGS_NEAREST_SIGMA) ? (o->sigma + 1.0 / theta) : theta;
        }
        if (o->compute_vectors) {
            s20_lift_ritz_vectors(V, Y_arrow, n, last_m_actual, last_take, sel_idx,
                                  result->eigenvectors);
        }
        result->n_converged = last_take;
        result->iterations = total_iters;
        result->residual_norm = last_partial_res;
    }

cleanup:
    free(V);
    free(alpha);
    free(beta);
    free(v0);
    free(residual_vec);
    free(T_arrow);
    free(theta_arrow);
    free(Y_arrow);
    free(sel_idx);
    free(V_locked_tmp);
    free(theta_locked_tmp);
    free(beta_coupling_tmp);
    lanczos_restart_state_free(&state);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════════
 * LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient
 *          (Sprint 21 Days 7-10; Knyazev 2001)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Day 7 lands the API surface, the design block below, and stubs
 * (`s21_lobpcg_orthonormalize_block`, `s21_lobpcg_rr_step`,
 * `s21_lobpcg_solve`) that return SPARSE_ERR_BADARG.  Day 8
 * implements the unpreconditioned core, Day 9 wires preconditioning
 * + the BLOPEX P-update, Day 10 tunes AUTO routing.
 *
 * Why a third backend.  Lanczos (with full reorth) is the workhorse
 * for symmetric eigenproblems on well-conditioned A — convergence is
 * geometric in the spectral gap and full-MGS reorth keeps the
 * Krylov basis orthonormal to machine precision.  Two regimes
 * motivate LOBPCG:
 *
 *   1. **Ill-conditioned SPD.** When `cond(A)` reaches 1e6+, the
 *      Lanczos spectral-gap rate slows to 1 − O(1/sqrt(cond)) per
 *      step.  A cheap preconditioner M ≈ A (IC(0), LDL^T) accelerates
 *      LOBPCG to an effective rate determined by `cond(M^{-1}·A)` —
 *      often four or five orders of magnitude faster on the same
 *      fixture.  Lanczos has no equivalent inner preconditioning
 *      hook (shift-invert is the closest analogue, but it requires
 *      a near-eigenvalue σ to work).
 *
 *   2. **Block convergence.** When the requested eigenvalues are
 *      clustered (k = 5 from a tightly-packed bottom of the
 *      spectrum), Lanczos converges them sequentially while LOBPCG
 *      converges them in parallel via the block_size > k mechanism.
 *      Each LOBPCG iteration costs O(block_size) matvecs vs Lanczos's
 *      one, so the parallel-block win has to be measured per
 *      problem; Day 10's AUTO threshold encodes the rule of thumb.
 *
 * Pipeline.  Each LOBPCG iteration maintains three n × block_size
 * matrices stored column-major:
 *
 *     X : current eigenvector approximations (init: random-deterministic)
 *     W : preconditioned residual            (M^{-1} · R when precond
 *                                             != NULL, else R itself)
 *     P : previous search direction          (init: 0; updated each step)
 *
 * The block Rayleigh-Ritz step concatenates these into an
 * n × (3·block_size) basis Q, orthonormalises it, forms the dense
 * symmetric Gram matrix
 *
 *     G = Q^T · A · Q
 *
 * (size 3·block_size × 3·block_size, computed by applying `op` to
 * each column of Q individually — Day 7 keeps the per-column
 * application; Sprint 22 may add a block matvec).  The dense
 * symmetric eigensolve via `s21_dense_sym_jacobi` (already in this
 * file from Day 2) yields all 3·block_size Ritz pairs (θ_j, y_j);
 * the block_size Ritz values matching `which` (LARGEST / SMALLEST /
 * NEAREST_SIGMA) become the new theta estimates, and the
 * combination coefficients produce the next X / P:
 *
 *     X_{new} = Q · Y[:, sel]                  (n × block_size)
 *     P_{new} = (W block of Q) · Y[W, sel]
 *             + (P block of Q) · Y[P, sel]     (n × block_size)
 *
 * Knyazev's "locally optimal" name comes from this Rayleigh-Ritz
 * minimisation over the 3-block subspace — at each step it picks the
 * best X / P combination available without lookahead, and the P
 * block carries the "momentum" that gives LOBPCG its
 * conjugate-gradient flavour.
 *
 * Numerical guards.
 *   - **Orthonormality.** Q must be orthonormal for G to be
 *     symmetric to within O(eps · cond) — `s21_lobpcg_orthonormalize_block`
 *     applies modified Gram-Schmidt with the Sprint 20 commit 70015a4
 *     scale-aware breakdown threshold; columns whose norm collapses
 *     below the threshold are ejected and the effective block size
 *     shrinks.  Day 8 unit tests that this ejection actually happens
 *     on near-singular Gram matrices.
 *   - **P-block stability.** Knyazev's original eq. 2.11 derives the
 *     new P from the difference of current and previous combination
 *     coefficients; on near-singular Gram matrices this loses
 *     significance.  Day 9 swaps in the BLOPEX (Stathopoulos 2007)
 *     formulation that conditions the P update on the Gram
 *     eigenvalue spread.  Day 8 ships the simpler Knyazev formula.
 *   - **Soft-locking.** Once a Ritz pair's residual drops below
 *     tol, optionally freeze it in X by setting the corresponding
 *     W and P columns to zero — saves work on the converged columns
 *     in subsequent iterations.  Day 9 wires this behind an opts
 *     flag (omitted from Day 7's struct since the field is
 *     deferrable; left for the Day 9 design step to land).
 *
 * Convergence gate.  Per-column residual `||R[:, j]||` scaled by
 * `max(|theta_j|, scale)` matches the Sprint 20 Wu/Simon convention
 * so `result->residual_norm` has consistent semantics across all
 * three backends.  When all `k` selected columns pass, emit and
 * return SPARSE_OK; on iteration-cap exhaustion, partial results
 * via SPARSE_ERR_NOT_CONVERGED.
 *
 * Spectrum modes.
 *   - LARGEST: standard LOBPCG converges to SMALLEST natively, so
 *     LARGEST wraps `op` in a negation `(neg_op)(x) := -A·x` and
 *     negates the returned eigenvalues.  Sprint 21 Day 10 lands the
 *     adapter; Day 7 stubs return BADARG for now.
 *   - SMALLEST: native LOBPCG path.  Day 8 implements this.
 *   - NEAREST_SIGMA: LOBPCG on `(A - σI)^{-1}` via the Sprint 20
 *     Day 12 LDL^T-shift-invert callback — the same `op_fn` /
 *     `op_ctx` setup `sparse_eigs_sym` builds for the Lanczos
 *     backends.  Post-process λ = σ + 1/θ.  Day 10 lands.
 *
 * Memory.  Peak `O((3·block_size + scratch) · n)` where scratch is
 * the dense Gram matrix and its eigenvector matrix (each
 * 3·block_size × 3·block_size) plus the per-iteration A·Q product
 * block (n × 3·block_size).  For block_size ≤ 30 this is ~5 MB on
 * bcsstk14 (n = 1806) — comparable to thick-restart's ~500 KB but
 * with much better convergence on ill-conditioned fixtures.
 *
 * References.
 *   - Knyazev, A. (2001).  Toward the Optimal Preconditioned
 *     Eigensolver: Locally Optimal Block Preconditioned Conjugate
 *     Gradient Method.  SIAM J. Sci. Comput. 23(2), 517-541.
 *   - Stathopoulos, A. (2007).  Nearly optimal preconditioned
 *     methods for Hermitian eigenproblems under limited memory.
 *     Part I: Seeking one eigenvalue.  SIAM J. Sci. Comput. 29(2),
 *     481-514.  (BLOPEX-style P-update formulation.)
 */

/* Day 8: orthonormalise an n × block_size_in column-major block via
 * per-column modified Gram-Schmidt with scale-aware breakdown
 * ejection.  Walks columns left-to-right, applying MGS against the
 * already-accepted columns 0..accepted-1 and either accepting (post-
 * MGS norm above threshold → normalise + advance) or ejecting
 * (post-MGS norm collapsed → linear-dependence on prior columns;
 * skip and forward-compact subsequent columns into this slot).
 *
 * Breakdown threshold mirrors the Sprint 20 commit 70015a4 pattern:
 * a relative `scale * 1e-14` threshold where `scale` is the running
 * max input column norm (pre-MGS), with a `DBL_MIN * 100` absolute
 * floor for the all-zero edge case. */
sparse_err_t s21_lobpcg_orthonormalize_block(double *Q, idx_t n, idx_t block_size_in,
                                             idx_t *block_size_out) {
    if (!Q || !block_size_out)
        return SPARSE_ERR_NULL;
    if (n < 1 || block_size_in < 0)
        return SPARSE_ERR_BADARG;

    if (block_size_in == 0) {
        *block_size_out = 0;
        return SPARSE_OK;
    }

    double scale = 0.0;
    idx_t accepted = 0;

    for (idx_t j = 0; j < block_size_in; j++) {
        double *col_in = Q + (size_t)j * (size_t)n;
        double *col_out = Q + (size_t)accepted * (size_t)n;

        /* Forward-compact when prior ejections have left a gap.
         * When `accepted == j` (no ejections yet) the copy is a
         * no-op and we skip it. */
        if (accepted != j) {
            for (idx_t i = 0; i < n; i++)
                col_out[i] = col_in[i];
        }

        /* Pre-MGS norm tracks the running scale of the input — a
         * "raw magnitude" reference for the post-MGS breakdown
         * comparison.  Without this, scale would always equal 1
         * (post-normalisation) and the threshold would fail to
         * distinguish a near-collinear column from a genuinely
         * orthogonal one. */
        double sq_in = 0.0;
        for (idx_t i = 0; i < n; i++)
            sq_in += col_out[i] * col_out[i];
        double norm_in = sqrt(sq_in);
        if (norm_in > scale)
            scale = norm_in;

        /* MGS against the prior accepted columns — reuses the
         * Sprint 20/21 OMP-parallel kernel so block orthogonalisation
         * automatically benefits from the matvec-class parallelism. */
        s21_mgs_reorth(col_out, Q, n, accepted);

        double sq_out = 0.0;
        for (idx_t i = 0; i < n; i++)
            sq_out += col_out[i] * col_out[i];
        double norm_out = sqrt(sq_out);

        double breakdown_tol = scale * 1e-14;
        if (breakdown_tol < DBL_MIN * 100.0)
            breakdown_tol = DBL_MIN * 100.0;

        if (norm_out > breakdown_tol) {
            double inv = 1.0 / norm_out;
            for (idx_t i = 0; i < n; i++)
                col_out[i] *= inv;
            accepted++;
        }
        /* Else: ejected.  Next column overwrites this slot via the
         * forward-compact copy at the top of the next iteration. */
    }

    *block_size_out = accepted;
    return SPARSE_OK;
}

/* Day 8: deterministic pseudo-random initial X for LOBPCG.  Extends
 * the Sprint 20 `s20_lanczos_starting_vector` per-column with a
 * column-dependent additive shift in the golden-ratio fractional
 * argument (using π for an irrational that is incommensurate with
 * the golden ratio, so different columns produce linearly-independent
 * starting vectors with high probability for the small n × bs we
 * care about).  Reproducible across runs. */
static void s21_lobpcg_init_X(double *X, idx_t n, idx_t bs) {
    for (idx_t j = 0; j < bs; j++) {
        double *col = X + (size_t)j * (size_t)n;
        for (idx_t i = 0; i < n; i++) {
            double x = (double)(i + 1) * 0.618033988749895 + (double)(j + 1) * 0.31415926535897932;
            col[i] = 0.3 + (x - floor(x));
        }
    }
}

/* Day 8 (Knyazev): one block Rayleigh-Ritz step.
 *
 *   Q ← [X | W | P]   (n × cap, where cap = 3·bs or 2·bs when P==NULL)
 *   orthonormalise Q in place                  → K_eff cols
 *   G ← Q^T · A · Q                            (K_eff × K_eff)
 *   diagonalise G via Jacobi                   → theta_full, Y
 *   select bs Ritz pairs per `which`           → sel_idx
 *   X_new ← Q · Y[:, sel_idx]                  (n × bs)
 *   P_new ← X_new − X · (X^T · X_new)          (orthogonal-projection
 *                                               formulation; equivalent
 *                                               to Knyazev eq. 2.11
 *                                               when X is orthonormal,
 *                                               which we maintain
 *                                               across iterations)
 *
 * Day 9 BLOPEX (Stathopoulos 2007) conditioning guard: when Jacobi
 * reports a near-singular G (smallest |theta_full| collapses
 * relative to the spectrum scale, indicating a rank-deficient Q
 * that the orthonormalise-block ejection didn't catch), fall back
 * to `P_new = 0` — equivalent to restarting the conjugate-gradient
 * direction track on the next iteration.  Rare in practice; observed
 * on extreme `bs > k` oversize-block runs against degenerate
 * spectra.  Same recovery BLOPEX 1.1 uses
 * (`lobpcg_solve_double.c::SolveTriDiagSystem`).  The orthogonal-
 * projection P-update formula is equivalent to BLOPEX's block-
 * structured `P_new = W'·Y_W + P'·Y_P` in exact arithmetic when X
 * is orthonormal — Day 9 keeps the cheaper Day 8 form and adds
 * only the conditioning guard, since the wholesale orthonormalise
 * pass already yields well-conditioned G in non-degenerate cases.
 *
 * Memory: n × cap doubles for Q + n × cap for AQ + cap × cap for G + Y
 * + cap doubles for theta_full + n × bs for X_new + n × bs for P_new.
 * For bs = 5 and n = 1806 this is ~75 KB per RR step. */
sparse_err_t s21_lobpcg_rr_step(lanczos_op_fn op, const void *ctx, idx_t n, idx_t block_size,
                                double *X, double *W, double *P, sparse_eigs_which_t which,
                                double *theta_out) {
    if (!op || !X || !W || !theta_out)
        return SPARSE_ERR_NULL;
    if (n < 1 || block_size < 1)
        return SPARSE_ERR_BADARG;

    int has_p = (P != NULL);
    idx_t cap = has_p ? 3 * block_size : 2 * block_size;
    if (cap > n)
        return SPARSE_ERR_BADARG;

    /* Workspace allocation.  All sizes bounded by (n, cap) where
     * cap ≤ 3·bs.  Single sparse_err_t propagation site below
     * via goto cleanup. */
    size_t nc_bytes = 0, cc_bytes = 0, nb_bytes = 0;
    if (size_mul_overflow((size_t)n, (size_t)cap, &nc_bytes) ||
        size_mul_overflow(nc_bytes, sizeof(double), &nc_bytes) ||
        size_mul_overflow((size_t)cap, (size_t)cap, &cc_bytes) ||
        size_mul_overflow(cc_bytes, sizeof(double), &cc_bytes) ||
        size_mul_overflow((size_t)n, (size_t)block_size, &nb_bytes) ||
        size_mul_overflow(nb_bytes, sizeof(double), &nb_bytes))
        return SPARSE_ERR_ALLOC;

    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *Q = malloc(nc_bytes);
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *AQ = malloc(nc_bytes);
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *G = malloc(cc_bytes);
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *Y = malloc(cc_bytes);
    double *theta_full = malloc((size_t)cap * sizeof(double));
    idx_t *sel_idx = malloc((size_t)block_size * sizeof(idx_t));
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *X_new = malloc(nb_bytes);
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *P_new = has_p ? malloc(nb_bytes) : NULL;

    sparse_err_t rc = SPARSE_OK;
    if (!Q || !AQ || !G || !Y || !theta_full || !sel_idx || !X_new || (has_p && !P_new)) {
        rc = SPARSE_ERR_ALLOC;
        goto cleanup;
    }

    /* Q ← [X | W | P] (column-major concatenation). */
    size_t nb = (size_t)n * (size_t)block_size;
    memcpy(Q, X, nb * sizeof(double));
    memcpy(Q + nb, W, nb * sizeof(double));
    if (has_p)
        memcpy(Q + 2 * nb, P, nb * sizeof(double));

    /* In-place orthonormalisation of Q with scale-aware breakdown
     * ejection.  K_eff ≤ cap is the effective subspace dimension. */
    idx_t K_eff = 0;
    sparse_err_t err = s21_lobpcg_orthonormalize_block(Q, n, cap, &K_eff);
    if (err != SPARSE_OK) {
        rc = err;
        goto cleanup;
    }
    if (K_eff < block_size) {
        /* The X block alone didn't survive orthogonalisation —
         * caller's X is rank-deficient.  Punt back to the outer
         * loop's allocation/init path. */
        rc = SPARSE_ERR_BADARG;
        goto cleanup;
    }

    /* AQ ← A · Q, column-by-column.  Sprint 22 may swap in a block
     * matvec; for now per-column matches the existing op_fn shape
     * unchanged from the Lanczos backends. */
    for (idx_t j = 0; j < K_eff; j++) {
        err = op(ctx, n, Q + (size_t)j * (size_t)n, AQ + (size_t)j * (size_t)n);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
    }

    /* G ← Q^T · AQ.  Symmetric K_eff × K_eff Gram matrix; explicit
     * symmetrisation suppresses the ~eps·||A|| asymmetry that
     * floating-point rounding leaves in the off-diagonal entries
     * (matters for Jacobi's `apq == aqp` invariant). */
    for (idx_t i = 0; i < K_eff; i++) {
        const double *qi = Q + (size_t)i * (size_t)n;
        for (idx_t j = 0; j < K_eff; j++) {
            const double *aqj = AQ + (size_t)j * (size_t)n;
            double s = 0.0;
            for (idx_t r = 0; r < n; r++)
                s += qi[r] * aqj[r];
            G[(size_t)i + (size_t)j * (size_t)K_eff] = s;
        }
    }
    for (idx_t i = 0; i < K_eff; i++) {
        for (idx_t j = i + 1; j < K_eff; j++) {
            size_t ij = (size_t)i + (size_t)j * (size_t)K_eff;
            size_t ji = (size_t)j + (size_t)i * (size_t)K_eff;
            double avg = 0.5 * (G[ij] + G[ji]);
            G[ij] = avg;
            G[ji] = avg;
        }
    }

    /* Diagonalise G via the Sprint 21 Day 2 dense Jacobi helper.
     * theta_full is sorted ascending; Y is K_eff × K_eff column-major. */
    err = s21_dense_sym_jacobi(G, K_eff, theta_full, Y);
    if (err != SPARSE_OK) {
        rc = err;
        goto cleanup;
    }

    /* Select block_size Ritz pairs per `which`. */
    idx_t take = s20_select_indices(theta_full, K_eff, which, block_size, sel_idx);
    if (take < block_size) {
        /* Subspace too small to extract block_size pairs.  Caller's
         * outer loop treats this as a non-convergent step. */
        rc = SPARSE_ERR_NOT_CONVERGED;
        goto cleanup;
    }

    /* Day 9 BLOPEX-style conditioning guard.  Detect a rank-deficient
     * Gram matrix that the orthonormalise-block ejection missed by
     * inspecting the spread of Jacobi's eigenvalues: when the
     * smallest |theta_full| collapses below `scale * 1e-12` relative
     * to the running max, treat the iteration as degenerate and
     * restart the CG direction (P_new = 0) on the next outer-loop
     * step rather than producing a numerically suspect P_new. */
    double scale_theta = 0.0;
    for (idx_t l = 0; l < K_eff; l++) {
        double a = fabs(theta_full[l]);
        if (a > scale_theta)
            scale_theta = a;
    }
    int gram_singular = 0;
    if (scale_theta > 0.0) {
        double cond_floor = scale_theta * 1e-12;
        for (idx_t l = 0; l < K_eff; l++) {
            if (fabs(theta_full[l]) < cond_floor) {
                gram_singular = 1;
                break;
            }
        }
    }

    /* X_new ← Q · Y[:, sel_idx].  Column j of X_new is a linear
     * combination of Q's columns weighted by Y's selected eigenvector. */
    for (idx_t j = 0; j < block_size; j++) {
        const double *yj = Y + (size_t)sel_idx[j] * (size_t)K_eff;
        double *xn = X_new + (size_t)j * (size_t)n;
        for (idx_t i = 0; i < n; i++)
            xn[i] = 0.0;
        for (idx_t c = 0; c < K_eff; c++) {
            double yc = yj[c];
            if (yc == 0.0)
                continue;
            const double *qc = Q + (size_t)c * (size_t)n;
            for (idx_t i = 0; i < n; i++)
                xn[i] += yc * qc[i];
        }
    }

    /* P_new ← X_new − X · (X^T · X_new) (Knyazev 2001 eq. 2.11
     * orthogonal-projection formulation).  When the conditioning
     * guard fires, fall back to P_new = 0 — equivalent to restarting
     * the CG track. */
    if (has_p) {
        if (gram_singular) {
            memset(P_new, 0, nb * sizeof(double));
        } else {
            for (idx_t j = 0; j < block_size; j++) {
                const double *xn = X_new + (size_t)j * (size_t)n;
                double *pn = P_new + (size_t)j * (size_t)n;
                for (idx_t i = 0; i < n; i++)
                    pn[i] = xn[i];
                for (idx_t l = 0; l < block_size; l++) {
                    const double *xl = X + (size_t)l * (size_t)n;
                    double dot = 0.0;
                    for (idx_t i = 0; i < n; i++)
                        dot += xn[i] * xl[i];
                    for (idx_t i = 0; i < n; i++)
                        pn[i] -= dot * xl[i];
                }
            }
        }
    }

    /* Write back X = X_new, P = P_new (when P provided), and the
     * selected Ritz values. */
    memcpy(X, X_new, nb * sizeof(double));
    if (has_p)
        memcpy(P, P_new, nb * sizeof(double));
    for (idx_t j = 0; j < block_size; j++)
        theta_out[j] = theta_full[sel_idx[j]];

cleanup:
    free(Q);
    free(AQ);
    free(G);
    free(Y);
    free(theta_full);
    free(sel_idx);
    free(X_new);
    free(P_new);
    return rc;
}

/* Day 8 + 9: LOBPCG outer loop.
 *
 * 1. Allocate X, R, W, P, AX (each n × bs).  R is the raw residual
 *    AX − X·diag(theta); W is the preconditioned residual that gets
 *    fed into the Rayleigh-Ritz step (W = R when `o->precond` is
 *    NULL — vanilla LOBPCG; W = precond(R) otherwise).
 * 2. Initialise X with deterministic pseudo-random columns,
 *    orthonormalise.
 * 3. Compute AX = A · X and initial Rayleigh quotients
 *    theta_j = <X[:, j], AX[:, j]>.
 * 4. Loop until convergence or max_iters:
 *    a. R = AX − X · diag(theta).
 *    b. Per-column Wu/Simon residual:
 *           ||R[:, j]|| / max(|theta_j|, scale)
 *       Track per-column convergence flags + max across columns.
 *    c. If all columns converged: emit and return.
 *    d. W ← precond(R) when o->precond != NULL, else W = R.
 *    e. Soft-locking (when o->lobpcg_soft_lock is set): for any
 *       per-column converged j, zero W[:, j] and P[:, j] before
 *       running the RR step — the orthonormaliser ejects those
 *       zero columns, so the active Rayleigh-Ritz subspace shrinks
 *       to (bs + active_W + active_P) instead of (3·bs).  The
 *       locked X[:, j] stays in the basis and its Ritz pair is
 *       preserved by the RR step (X is in Q, so A·X[:, j] =
 *       theta[j]·X[:, j] + O(residual_j) maps back to the same
 *       column of Y).
 *    f. Run `s21_lobpcg_rr_step(X, W, P)` — updates X, P, theta
 *       in place via the BLOPEX-style block-preserving Rayleigh-
 *       Ritz pipeline.
 *    g. Recompute AX = A · X for the next residual.
 *
 * On the first iteration P is passed as NULL (signals "no P yet");
 * `s21_lobpcg_rr_step` handles the 2·bs subspace case by skipping
 * the P concatenation and the P_new write-back.  Subsequent
 * iterations pass the persistent P buffer.
 *
 * Preconditioning convergence claim (Day 9 PLAN target): on an
 * ill-conditioned SPD with cond(A) ~ 1e6, vanilla LOBPCG converges
 * in O(sqrt(cond(A)) / log(1/eps)) iterations while preconditioned
 * LOBPCG with M ≈ A converges in O(sqrt(cond(M^{-1}·A)) / log(1/eps))
 * — typically 5×+ faster.  The Day 9 regression tests verify this
 * with IC(0) and LDL^T preconditioners on a 1D-Laplacian-shifted
 * fixture.
 *
 * On reaching `max_iters` without convergence, returns
 * SPARSE_ERR_NOT_CONVERGED with the current best Ritz values written
 * to result->eigenvalues (matching the Sprint 20 grow-m partial-
 * results contract). */
sparse_err_t s21_lobpcg_solve(lanczos_op_fn op, const void *ctx, idx_t n, idx_t k,
                              const sparse_eigs_opts_t *o, double eff_tol, idx_t max_iters,
                              sparse_eigs_t *result) {
    if (!op || !o || !result || !result->eigenvalues)
        return SPARSE_ERR_NULL;
    if (n < 1 || k < 1 || max_iters < 1)
        return SPARSE_ERR_BADARG;

    /* Resolve effective block size.  o->block_size == 0 selects the
     * library default `bs = k`; sparse_eigs_sym already validated
     * that nonzero values satisfy `k ≤ block_size ≤ n`. */
    idx_t bs = (o->block_size > 0) ? o->block_size : k;
    if (bs > n)
        bs = n;
    if (bs < k)
        return SPARSE_ERR_BADARG;

    /* peak_basis_size telemetry: outer-loop holds X + W + P + AX = 4·bs
     * length-n vectors live; the RR step adds another 3·bs (Q) +
     * 3·bs (AQ) inside its scope.  Peak across the call is therefore
     * 4·bs (outer) + 6·bs (RR transient) = 10·bs.  Lower bound for
     * the reservation comparison vs the Lanczos backends. */
    result->peak_basis_size = 10 * bs;

    /* Workspace allocation.  Single sparse_err_t propagation via
     * goto cleanup; mirrors the thick-restart outer loop's pattern. */
    size_t nb_bytes = 0;
    if (size_mul_overflow((size_t)n, (size_t)bs, &nb_bytes) ||
        size_mul_overflow(nb_bytes, sizeof(double), &nb_bytes))
        return SPARSE_ERR_ALLOC;

    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *X = malloc(nb_bytes);
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *R = malloc(nb_bytes); /* raw residual AX − X·diag(theta) */
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *W = malloc(nb_bytes); /* preconditioned residual fed into RR */
    double *P = NULL;             /* allocated lazily after the first RR step */
    // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
    double *AX = malloc(nb_bytes);
    double *theta = malloc((size_t)bs * sizeof(double));
    int *converged = calloc((size_t)bs, sizeof(int));

    sparse_err_t rc = SPARSE_ERR_NOT_CONVERGED;
    idx_t total_iters = 0;
    double last_res_rel = 0.0;

    if (!X || !R || !W || !AX || !theta || !converged) {
        rc = SPARSE_ERR_ALLOC;
        goto cleanup;
    }

    /* Step 1: deterministic pseudo-random init + orthonormalise. */
    s21_lobpcg_init_X(X, n, bs);
    idx_t bs_eff = 0;
    sparse_err_t err = s21_lobpcg_orthonormalize_block(X, n, bs, &bs_eff);
    if (err != SPARSE_OK) {
        rc = err;
        goto cleanup;
    }
    if (bs_eff < k) {
        /* Pathological: starting vectors collapsed to a subspace
         * smaller than k.  Should never happen with the golden-
         * ratio init for n ≥ k ≥ 1. */
        rc = SPARSE_ERR_BADARG;
        goto cleanup;
    }
    bs = bs_eff;

    /* Step 2: AX = A · X, then theta_j = <X[:, j], AX[:, j]>. */
    for (idx_t j = 0; j < bs; j++) {
        err = op(ctx, n, X + (size_t)j * (size_t)n, AX + (size_t)j * (size_t)n);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }
    }
    for (idx_t j = 0; j < bs; j++) {
        const double *xj = X + (size_t)j * (size_t)n;
        const double *axj = AX + (size_t)j * (size_t)n;
        double s = 0.0;
        for (idx_t i = 0; i < n; i++)
            s += xj[i] * axj[i];
        theta[j] = s;
    }

    /* Step 3: outer loop until convergence or max_iters. */
    for (idx_t iter = 0; iter < max_iters; iter++) {
        total_iters = iter + 1;

        /* Raw residual R = AX − X · diag(theta).  Convergence gate
         * runs against R (the un-preconditioned residual) so the
         * tolerance has problem-physical meaning regardless of the
         * preconditioner choice. */
        for (idx_t j = 0; j < bs; j++) {
            const double *xj = X + (size_t)j * (size_t)n;
            const double *axj = AX + (size_t)j * (size_t)n;
            double *rj = R + (size_t)j * (size_t)n;
            double tj = theta[j];
            for (idx_t i = 0; i < n; i++)
                rj[i] = axj[i] - tj * xj[i];
        }

        /* Per-column Wu/Simon residual: ||R[:, j]|| / max(|theta_j|, scale).
         * scale = max |theta| across the block; matches the Lanczos
         * backends so result->residual_norm has consistent semantics. */
        double scale = 0.0;
        for (idx_t j = 0; j < bs; j++) {
            double a = fabs(theta[j]);
            if (a > scale)
                scale = a;
        }
        double max_res_rel = 0.0;
        idx_t n_locked = 0;
        for (idx_t j = 0; j < bs; j++) {
            const double *rj = R + (size_t)j * (size_t)n;
            double sq = 0.0;
            for (idx_t i = 0; i < n; i++)
                sq += rj[i] * rj[i];
            double r_norm = sqrt(sq);
            double anchor = fabs(theta[j]);
            if (anchor < scale * 1e-12)
                anchor = scale > 0.0 ? scale : 1.0;
            double rel = r_norm / anchor;
            if (rel > max_res_rel)
                max_res_rel = rel;
            /* Per-column convergence flag.  Once a column meets tol,
             * stays converged across iterations (soft-lock semantics
             * — matches the BLOPEX reference: latched flag, not
             * recomputed-from-scratch). */
            if (rel <= eff_tol)
                converged[j] = 1;
            if (converged[j])
                n_locked++;
        }
        last_res_rel = max_res_rel;
        if (max_res_rel <= eff_tol) {
            rc = SPARSE_OK;
            break;
        }

        /* Day 9: preconditioning.  W ← M^{-1} · R when o->precond is
         * non-NULL; else W = R (vanilla LOBPCG).  The precond is
         * applied per-column unconditionally; soft-locking (below)
         * then zeroes locked columns regardless of whether they
         * passed through the precond. */
        if (o->precond) {
            for (idx_t j = 0; j < bs; j++) {
                const double *rj = R + (size_t)j * (size_t)n;
                double *wj = W + (size_t)j * (size_t)n;
                err = o->precond(o->precond_ctx, n, rj, wj);
                if (err != SPARSE_OK) {
                    rc = err;
                    goto cleanup;
                }
            }
        } else {
            memcpy(W, R, nb_bytes);
        }

        /* Soft-locking (Day 9): when enabled and a column has
         * converged, zero its W and (if allocated) P entries so the
         * RR step's orthonormaliser ejects them.  Locked X[:, j]
         * stays in the basis; the RR step preserves its Ritz pair
         * because X is part of Q's leading bs columns. */
        if (o->lobpcg_soft_lock && n_locked > 0) {
            for (idx_t j = 0; j < bs; j++) {
                if (!converged[j])
                    continue;
                memset(W + (size_t)j * (size_t)n, 0, (size_t)n * sizeof(double));
                if (P)
                    memset(P + (size_t)j * (size_t)n, 0, (size_t)n * sizeof(double));
            }
        }

        /* Rayleigh-Ritz step.  P is NULL on the first iteration so
         * the RR step works on the 2·bs subspace [X | W]; from
         * iter 1 onward P is allocated and carries the search
         * direction across iterations. */
        err = s21_lobpcg_rr_step(op, ctx, n, bs, X, W, P, o->which, theta);
        if (err != SPARSE_OK) {
            rc = err;
            goto cleanup;
        }

        /* Lazy P allocation.  After the first RR step X has been
         * updated in span(X_old, W) but P_new wasn't computed (P was
         * NULL).  Allocate P and zero it for the next iteration's
         * RR step, which then builds the BLOPEX P_new internally. */
        if (!P) {
            // NOLINTNEXTLINE(clang-analyzer-optin.portability.UnixAPI)
            P = malloc(nb_bytes);
            if (!P) {
                rc = SPARSE_ERR_ALLOC;
                goto cleanup;
            }
            memset(P, 0, nb_bytes);
        }

        /* AX = A · X for the next residual. */
        for (idx_t j = 0; j < bs; j++) {
            err = op(ctx, n, X + (size_t)j * (size_t)n, AX + (size_t)j * (size_t)n);
            if (err != SPARSE_OK) {
                rc = err;
                goto cleanup;
            }
        }
    }

    /* Emit results.  Pick the first k of the bs converged columns
     * (the RR step's selection has already ordered them per `which`). */
    idx_t emit = (k < bs) ? k : bs;
    for (idx_t j = 0; j < emit; j++) {
        double t = theta[j];
        result->eigenvalues[j] = (o->which == SPARSE_EIGS_NEAREST_SIGMA) ? (o->sigma + 1.0 / t) : t;
    }
    if (o->compute_vectors) {
        for (idx_t j = 0; j < emit; j++) {
            const double *xj = X + (size_t)j * (size_t)n;
            double *vj = result->eigenvectors + (size_t)j * (size_t)n;
            for (idx_t i = 0; i < n; i++)
                vj[i] = xj[i];
        }
    }
    result->n_converged = (rc == SPARSE_OK) ? emit : 0;
    result->iterations = total_iters;
    result->residual_norm = last_res_rel;

cleanup:
    free(X);
    free(R);
    free(W);
    free(P);
    free(AX);
    free(theta);
    free(converged);
    return rc;
}
