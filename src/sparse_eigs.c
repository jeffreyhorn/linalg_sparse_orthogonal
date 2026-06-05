/* Request POSIX clock_gettime on platforms that gate it behind
 * _POSIX_C_SOURCE; Windows uses timespec_get below instead. */
#if !defined(_WIN32) && (!defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 199309L)
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _POSIX_C_SOURCE 199309L
#endif

/**
 * @file sparse_eigs.c
 * @brief Sparse symmetric eigensolver front door plus shared Lanczos-family
 *        helpers.
 *
 * This file owns the public entry points, backend selection, shared Lanczos
 * kernels, shift-invert setup, dense Jacobi helper, and refinement logic.
 * Backend-specific implementations that grew large enough to stand on their
 * own live in `src/sparse_eigs_thick_restart.c` and
 * `src/sparse_eigs_lobpcg.c`.
 */

#include "sparse_eigs.h"

#include "sparse_alloc_internal.h"
#include "sparse_dense.h"
#include "sparse_eigs_internal.h"
#include "sparse_eigs_workspace_internal.h"
#include "sparse_ldlt.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "sparse_vector.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Shared monotonic timer for progress reporting and benchmark-style
 * elapsed-time bookkeeping. */
double s29_eigs_now_s(void) {
    struct timespec ts;
#ifdef _WIN32
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

#ifdef SPARSE_OPENMP
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <omp.h>
#pragma GCC diagnostic pop
#endif

/* ─── Shared MGS Reorth Kernel ─────────────────────────────────────── */

/* Small-n gate for the MGS reorth parallelism.  Below this `n`,
 * per-call OMP fork/join overhead (measured ~5-20 μs on macOS
 * Homebrew libomp) dominates the dot-product + daxpy work, and
 * the parallel reduction is a net loss on small matrices, while
 * larger SuiteSparse-style problems do benefit.  `500` is a
 * conservative crossover that keeps the small cases on the serial
 * path and still enables parallelism well before the large-matrix
 * regime.  Callers who need a different
 * crossover on their hardware can override by building with
 * `-DSPARSE_EIGS_OMP_REORTH_MIN_N=<value>`.  See
 * the project performance notes for the threshold rationale. */
#ifndef SPARSE_EIGS_OMP_REORTH_MIN_N
#define SPARSE_EIGS_OMP_REORTH_MIN_N 500
#endif

/* s21_mgs_reorth: orthogonalise `w` against each of V[:, 0..k_stored-1]
 * via classical modified Gram-Schmidt (MGS).  The outer `j` loop
 * is serial — MGS stability requires each iteration to see the
 * partially-orthogonalised `w` from the previous subtraction
 * (classical Gram-Schmidt parallelises `j` but loses the stability
 * bound; the inner dot-product and daxpy bodies are independent
 * across `i` and parallelised under `-DSPARSE_OPENMP`, using the
 * same pragma pattern as `sparse_matvec`.
 *
 * The `if (n >= SPARSE_EIGS_OMP_REORTH_MIN_N)` clause causes each
 * `parallel for` to run serially (a single
 * implicit team of one) when `n` is small enough that OMP
 * overhead would exceed the parallel work.  The clause is a
 * zero-cost no-op in serial builds (the whole pragma is
 * `#ifdef SPARSE_OPENMP`-gated out).
 *
 * Serial builds are bit-for-bit identical to the inline MGS body
 * this helper replaced.  Centralising the kernel keeps the standard
 * Lanczos path and the thick-restart path in sync.
 *
 * Shared across `lanczos_iterate_op` and
 * `lanczos_thick_restart_iterate`.  Does nothing when
 * `k_stored == 0`. */
void s21_mgs_reorth(double *w, const double *V, idx_t n, idx_t k_stored) {
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

/* ═══════════════════════════════════════════════════════════════════════
 * Lanczos — 3-term recurrence + optional reorthogonalization
 * ═══════════════════════════════════════════════════════════════════════
 *
 * `lanczos_iterate` builds an m-step Lanczos basis V and
 * tridiagonal T from a symmetric matrix A and starting vector v0.
 * The optional `reorthogonalize` gate layers full MGS reorthogonalization
 * on top of the classical three-term recurrence.
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
 * sharpest at the extremes of the spectrum, which is why the solver
 * selects `LARGEST` / `SMALLEST` Ritz values from T.
 *
 * Finite-precision caveat.  Without reorthogonalization V^T·V
 * drifts from I as k grows and "ghost" Ritz values — duplicates of
 * eigenvalues that already converged — appear in T's spectrum.
 * Paige's analysis (1972) shows ghosts arrive around k ≈ condition-
 * number-scale steps.  Full reorth suppresses this; the basic
 * recurrence is sufficient for the unit tests below where m
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
 * Reorthogonalization.  When the caller sets `reorthogonalize != 0`,
 * after the standard 3-term recurrence
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
 * inputs at 2× the reorth cost.  Not currently wired; if future
 * convergence tests show lingering orthogonality drift, add a
 * `opts->reorthogonalize == 2` escalation.
 *
 * Unit tests via `sparse_eigs_internal.h` exercise both paths
 * through `lanczos_iterate` directly. */

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
    size_t n_size = 0;
    size_t w_bytes = 0;
    if (sparse_idx_to_size_checked(n, &n_size) ||
        sparse_size_mul_overflow(n_size, sizeof(double), &w_bytes))
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
         * path or (A - sigma*I)^{-1} via LDL^T solve for shift-invert
         * mode. */
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
         * MGS from classical Gram-Schmidt.  The shared
         * `s21_mgs_reorth` helper keeps this path and the
         * thick-restart path aligned. */
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
 * Grow-m Lanczos outer loop
 * ═══════════════════════════════════════════════════════════════════════
 *
 * This outer-loop driver:
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
 *      is itself the convergence signal.
 *   5. Selects the `k` values matching `opts->which` (LARGEST
 *      from the top of the ascending θ array; SMALLEST from the
 *      bottom) and emits.
 *   6. On non-convergence, restarts: use the last Lanczos vector
 *      of the longer run (best approximation of the extreme Ritz
 *      direction) as the next v0.  Extends the iteration budget
 *      and retries.  This path intentionally stays simpler than the
 *      full arrowhead-state thick-restart backend.
 *
 * This grow-m path is intentionally conservative: it relies on a
 * stability proxy rather than Wu/Simon per-pair residual estimators
 * and trades memory for a smaller control surface. */

/* Deterministic starting vector — golden-ratio fractional mixing.
 * Avoids alignment with any standard-basis eigenvector (diagonal
 * fixtures would otherwise terminate Lanczos in one step) and is
 * reproducible across runs. */
void s20_lanczos_starting_vector(double *v0, idx_t n) {
    for (idx_t i = 0; i < n; i++) {
        double x = (double)(i + 1) * 0.618033988749895;
        v0[i] = 0.3 + (x - floor(x));
    }
}

/* Quick approximation of ||A||_inf via scaling the final beta
 * against the largest Ritz value.  If theta_max dominates the
 * iteration's spectrum, ||A||_inf is at least |theta_max|; we
 * use that as the tolerance anchor. */
double s20_spectrum_scale(const double *theta, idx_t m) {
    double s = 0.0;
    for (idx_t i = 0; i < m; i++) {
        double a = fabs(theta[i]);
        if (a > s)
            s = a;
    }
    return s;
}

/* Ritz-pair extraction — returns eigenvalues of the tridiagonal T
 * in `theta_out` (ascending) and T's orthonormal eigenvectors as
 * columns of `Y_out` (m × m, column-major) so the later lift
 * V · Y[:, j] produces full-problem Ritz vectors.
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

/* Shift-invert Lanczos operator.  `ctx` is a `(const sparse_ldlt_t *)`
 * pointing at a pre-computed LDL^T
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
idx_t s20_select_indices(const double *theta, idx_t m, sparse_eigs_which_t which, idx_t k_want,
                         idx_t *sel_idx) {
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
 * V's columns are already orthonormal (assuming full
 * reorthogonalization) so the lifted vectors inherit unit norm up
 * to the MGS
 * drift bound (‖ε‖ ≲ 1e-12 on well-conditioned A).  Ritz vectors of
 * (A - σI)^{-1} are also eigenvectors of A (same eigenspaces), so
 * the same lift works for shift-invert mode. */
void s20_lift_ritz_vectors(const double *V, const double *Y, idx_t n, idx_t m, idx_t take,
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

/* Eigenpair refinement via Rayleigh-quotient iteration.  Each
 * converged pair `(lambda_j, v_j)` is refined by:
 *
 *   for iter in [0, max_iters):
 *     r = A * v_j - lambda_j * v_j
 *     if ||r||_2 / max(|lambda_j|, 1) < TIGHT_TOL: break
 *     factor: (A - lambda_j * I) = L*D*L^T          via sparse_ldlt_factor_opts
 *     solve:  (A - lambda_j * I) * y = v_j           via sparse_ldlt_solve
 *     v_j   := y / ||y||_2
 *     lambda_j := v_j^T A v_j                        (Rayleigh quotient)
 *
 * Edge cases:
 *   - Singular `(A - lambda_j * I)` (lambda_j matches an eigenvalue to
 *     full precision): perturb shift by `100 * eps * max(|lambda_j|, 1)`
 *     and retry once.  Two failures → stop refining this pair (likely
 *     already at the eigenvalue).
 *   - Rayleigh-quotient stall (|lambda_j_new - lambda_j_prev| below
 *     TIGHT_TOL): break early.
 *   - Degenerate solve output (||y|| ~= 0): stop refining this pair.
 *
 * Updates `result->eigenvalues[j]` + `result->eigenvectors[:, j]` in
 * place + recomputes `result->residual_norm` as the max post-
 * refinement relative residual across converged pairs. */
#define SPARSE_EIGS_REFINE_TIGHT_TOL 1e-14
#define SPARSE_EIGS_REFINE_DEFAULT_MAX_ITERS 5

static double s29_refine_anchor(double lambda) {
    /* Anchor for relative residual / shift-perturbation / stall-break:
     * `max(|lambda|, 1.0)` per the Day-4 design contract.  The
     * lower-bounded `1.0` keeps the relative-residual criterion
     * `||r|| / anchor < TIGHT_TOL` from becoming unrealistically strict
     * for small-but-nonzero eigenvalues (e.g. |lambda| = 1e-6 would
     * otherwise demand ||r|| < 1e-20, which is below the
     * representation floor of double and would force refinement to
     * run to `refine_max_iters` with no benefit). */
    double abs_lambda = fabs(lambda);
    return (abs_lambda > 1.0) ? abs_lambda : 1.0;
}

static sparse_err_t s29_refine_pair(const SparseMatrix *A, idx_t n, double *v_j,
                                    double *lambda_j_io, idx_t max_iters, double *Av_scratch,
                                    double *y_scratch) {
    double lambda_j = *lambda_j_io;
    double prev_lambda = lambda_j;

    for (idx_t iter = 0; iter < max_iters; iter++) {
        /* Residual check: ||A v - lambda v|| / max(|lambda|, 1). */
        sparse_err_t mv_err = sparse_matvec(A, v_j, Av_scratch);
        if (mv_err != SPARSE_OK)
            return mv_err;
        double res_sq = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double r = Av_scratch[i] - lambda_j * v_j[i];
            res_sq += r * r;
        }
        double r_norm = sqrt(res_sq);
        double rel_res = r_norm / s29_refine_anchor(lambda_j);
        if (rel_res < SPARSE_EIGS_REFINE_TIGHT_TOL)
            break;

        /* Build A - shift * I.  Try the unperturbed shift first; on
         * SPARSE_ERR_SINGULAR retry with a small perturbation. */
        int factored = 0;
        sparse_ldlt_t ldlt = {0};
        double shift = lambda_j;
        for (int retry = 0; retry < 2 && !factored; retry++) {
            SparseMatrix *A_shifted = sparse_copy(A);
            if (!A_shifted)
                return SPARSE_ERR_ALLOC;
            if (retry > 0) {
                double eps = 2.2204460492503131e-16;
                double delta = 100.0 * eps * s29_refine_anchor(lambda_j);
                shift = lambda_j + delta;
            }
            for (idx_t i = 0; i < n; i++) {
                double dii = sparse_get(A_shifted, i, i);
                sparse_err_t serr = sparse_set(A_shifted, i, i, dii - shift);
                if (serr != SPARSE_OK) {
                    sparse_free(A_shifted);
                    return serr;
                }
            }
            sparse_ldlt_opts_t ldlt_opts = {
                .reorder = SPARSE_REORDER_NONE,
                .tol = 0.0,
                .backend = SPARSE_LDLT_BACKEND_AUTO,
            };
            sparse_err_t f_err = sparse_ldlt_factor_opts(A_shifted, &ldlt_opts, &ldlt);
            sparse_free(A_shifted);
            if (f_err == SPARSE_OK) {
                factored = 1;
            } else if (f_err == SPARSE_ERR_SINGULAR) {
                sparse_ldlt_free(&ldlt);
                /* retry with perturbed shift */
            } else {
                sparse_ldlt_free(&ldlt);
                return f_err;
            }
        }
        if (!factored)
            break; /* both shifts singular — stop refining this pair */

        /* Solve (A - shift * I) y = v_j. */
        sparse_err_t s_err = sparse_ldlt_solve(&ldlt, v_j, y_scratch);
        sparse_ldlt_free(&ldlt);
        if (s_err != SPARSE_OK)
            return s_err;

        /* v_j := y / ||y||. */
        double y_norm_sq = 0.0;
        for (idx_t i = 0; i < n; i++)
            y_norm_sq += y_scratch[i] * y_scratch[i];
        double y_norm = sqrt(y_norm_sq);
        if (y_norm < 1e-300)
            break; /* degenerate */
        double inv = 1.0 / y_norm;
        for (idx_t i = 0; i < n; i++)
            v_j[i] = y_scratch[i] * inv;

        /* Rayleigh quotient: lambda_j := v_j^T A v_j. */
        mv_err = sparse_matvec(A, v_j, Av_scratch);
        if (mv_err != SPARSE_OK)
            return mv_err;
        double rq = 0.0;
        for (idx_t i = 0; i < n; i++)
            rq += v_j[i] * Av_scratch[i];
        lambda_j = rq;

        /* Stall check. */
        double dl = fabs(lambda_j - prev_lambda);
        if (dl < SPARSE_EIGS_REFINE_TIGHT_TOL * s29_refine_anchor(lambda_j))
            break;
        prev_lambda = lambda_j;
    }

    *lambda_j_io = lambda_j;
    return SPARSE_OK;
}

static sparse_err_t s29_refine_eigenpairs(const SparseMatrix *A, const sparse_eigs_opts_t *opts,
                                          sparse_eigs_t *result) {
    if (!opts->refine || !opts->compute_vectors || result->n_converged <= 0)
        return SPARSE_OK;

    idx_t n = sparse_rows(A);
    idx_t max_iters =
        opts->refine_max_iters > 0 ? opts->refine_max_iters : SPARSE_EIGS_REFINE_DEFAULT_MAX_ITERS;

    double *Av = malloc((size_t)n * sizeof(double));
    double *y = malloc((size_t)n * sizeof(double));
    if (!Av || !y) {
        free(Av);
        free(y);
        return SPARSE_ERR_ALLOC;
    }

    double max_rel_res = 0.0;
    sparse_err_t rc = SPARSE_OK;
    for (idx_t j = 0; j < result->n_converged; j++) {
        double *v_j = result->eigenvectors + (size_t)j * (size_t)n;
        double lambda_j = result->eigenvalues[j];

        sparse_err_t pe = s29_refine_pair(A, n, v_j, &lambda_j, max_iters, Av, y);
        if (pe != SPARSE_OK) {
            rc = pe;
            break;
        }
        result->eigenvalues[j] = lambda_j;

        /* Recompute final residual for the residual_norm update. */
        sparse_err_t mv_err = sparse_matvec(A, v_j, Av);
        if (mv_err != SPARSE_OK) {
            rc = mv_err;
            break;
        }
        double res_sq = 0.0;
        for (idx_t i = 0; i < n; i++) {
            double r = Av[i] - lambda_j * v_j[i];
            res_sq += r * r;
        }
        double r_norm = sqrt(res_sq);
        double rel = r_norm / s29_refine_anchor(lambda_j);
        if (rel > max_rel_res)
            max_rel_res = rel;
    }

    if (rc == SPARSE_OK)
        result->residual_norm = max_rel_res;

    free(Av);
    free(y);
    return rc;
}

/* Helper: when `opts->refine` is requested, call the Day-5 refinement
 * post-pass and fold its return code into the outgoing rc.  We refine
 * on SPARSE_OK and SPARSE_ERR_NOT_CONVERGED — partial convergence
 * still has `result->n_converged` triplets in the output buffers that
 * the caller can tighten.  Refinement-step failures (allocation, solve)
 * override the backend's rc; success preserves the backend's rc. */
static sparse_err_t s29_maybe_refine(const SparseMatrix *A, const sparse_eigs_opts_t *opts,
                                     sparse_eigs_t *result, sparse_err_t backend_rc) {
    if (!opts->refine)
        return backend_rc;
    if (backend_rc != SPARSE_OK && backend_rc != SPARSE_ERR_NOT_CONVERGED)
        return backend_rc;
    sparse_err_t ref_rc = s29_refine_eigenpairs(A, opts, result);
    if (ref_rc != SPARSE_OK)
        return ref_rc;
    return backend_rc;
}

static sparse_eigs_opts_t s46_default_public_opts(void) {
    return (sparse_eigs_opts_t){
        .which = SPARSE_EIGS_LARGEST,
        .sigma = 0.0,
        .max_iterations = 0,
        .tol = 0.0,
        .reorthogonalize = 1,
        .compute_vectors = 0,
        .backend = SPARSE_EIGS_BACKEND_AUTO,
        .lobpcg_soft_lock = 1,
    };
}

static sparse_eigs_workspace_t *s49_eigs_handle_workspace(const sparse_eigs_handle_t *handle) {
    return handle ? (sparse_eigs_workspace_t *)handle->internal_state : NULL;
}

static sparse_eigs_backend_t s46_select_backend(idx_t n, idx_t k, const sparse_eigs_opts_t *o);

static sparse_err_t s49_eigs_handle_ensure(sparse_eigs_handle_t *handle,
                                           sparse_eigs_workspace_t **workspace_out) {
    if (!handle || !workspace_out)
        return SPARSE_ERR_NULL;

    sparse_eigs_workspace_t *workspace = s49_eigs_handle_workspace(handle);
    if (!workspace) {
        workspace = NULL;
        sparse_err_t err = sparse_malloc_array(1, sizeof(*workspace), (void **)&workspace);
        if (err != SPARSE_OK)
            return err;
        sparse_eigs_workspace_init(workspace);
        handle->internal_state = workspace;
    }

    *workspace_out = workspace;
    return SPARSE_OK;
}

static sparse_err_t s49_eigs_effective_max_iters(idx_t n, idx_t k, const sparse_eigs_opts_t *o,
                                                 idx_t *max_iters_out) {
    if (!max_iters_out)
        return SPARSE_ERR_NULL;

    if (o->max_iterations > 0) {
        int64_t min_required = (int64_t)2 * (int64_t)k + 10;
        if (min_required > (int64_t)n)
            min_required = (int64_t)n;
        if ((int64_t)o->max_iterations < min_required)
            return SPARSE_ERR_BADARG;
        *max_iters_out = o->max_iterations;
        return SPARSE_OK;
    }

    int64_t def_iters = (int64_t)10 * (int64_t)k + 20;
    if (def_iters < 100)
        def_iters = 100;
    if (def_iters > (int64_t)INT32_MAX)
        def_iters = (int64_t)INT32_MAX;
    *max_iters_out = (idx_t)def_iters;
    return SPARSE_OK;
}

static idx_t s49_eigs_growm_capacity(idx_t n, idx_t k, idx_t max_iters) {
    idx_t m_min = (n >= 2) ? 2 : n;
    idx_t m_cap = max_iters < n ? max_iters : n;
    int64_t m_cap_min = (int64_t)2 * (int64_t)k + 10;
    if ((int64_t)m_cap < m_cap_min)
        m_cap = (m_cap_min > (int64_t)n) ? n : (idx_t)m_cap_min;
    if (m_cap > n)
        m_cap = n;
    if (m_cap < m_min)
        m_cap = m_min;
    return m_cap;
}

static idx_t s49_eigs_thick_restart_capacity(idx_t n, idx_t k, idx_t max_iters) {
    int64_t m_restart_wide = (int64_t)2 * (int64_t)k + 20;
    if (m_restart_wide > (int64_t)n)
        m_restart_wide = (int64_t)n;
    if (m_restart_wide > (int64_t)max_iters)
        m_restart_wide = (int64_t)max_iters;
    if (m_restart_wide < (int64_t)k + 1)
        m_restart_wide = (int64_t)k + 1;
    if (m_restart_wide > (int64_t)n)
        m_restart_wide = (int64_t)n;
    return (idx_t)m_restart_wide;
}

static sparse_err_t s49_eigs_handle_prepare_backend(sparse_eigs_workspace_t *workspace, idx_t n,
                                                    idx_t k, const sparse_eigs_opts_t *o,
                                                    idx_t max_iters) {
    sparse_eigs_backend_t backend = s46_select_backend(n, k, o);
    switch (backend) {
    case SPARSE_EIGS_BACKEND_LOBPCG: {
        idx_t bs = (o->block_size > 0) ? o->block_size : k;
        if (bs > n)
            bs = n;
        sparse_eigs_lobpcg_workspace_view_t view;
        return sparse_eigs_workspace_prepare_lobpcg(workspace, n, bs, 1, &view);
    }
    case SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART: {
        idx_t m_restart = s49_eigs_thick_restart_capacity(n, k, max_iters);
        sparse_eigs_thick_restart_workspace_view_t view;
        return sparse_eigs_workspace_prepare_thick_restart(workspace, n, m_restart, k, &view);
    }
    case SPARSE_EIGS_BACKEND_LANCZOS:
    case SPARSE_EIGS_BACKEND_AUTO: {
        idx_t m_cap = s49_eigs_growm_capacity(n, k, max_iters);
        sparse_eigs_growm_workspace_view_t view;
        return sparse_eigs_workspace_prepare_growm(workspace, n, m_cap, k, &view);
    }
    }

    return SPARSE_ERR_BADARG;
}

void sparse_eigs_handle_init(sparse_eigs_handle_t *handle) {
    if (handle)
        *handle = (sparse_eigs_handle_t){0};
}

void sparse_eigs_handle_free(sparse_eigs_handle_t *handle) {
    if (!handle)
        return;
    sparse_eigs_workspace_t *workspace = s49_eigs_handle_workspace(handle);
    if (workspace) {
        sparse_eigs_workspace_free(workspace);
        free(workspace);
    }
    *handle = (sparse_eigs_handle_t){0};
}

sparse_err_t sparse_eigs_handle_prepare(sparse_eigs_handle_t *handle, idx_t n, idx_t k,
                                        const sparse_eigs_opts_t *opts) {
    const sparse_eigs_opts_t defaults = s46_default_public_opts();
    const sparse_eigs_opts_t *o = opts ? opts : &defaults;
    if (n < 1 || k < 1 || k > n)
        return SPARSE_ERR_BADARG;
    if (o->which != SPARSE_EIGS_LARGEST && o->which != SPARSE_EIGS_SMALLEST &&
        o->which != SPARSE_EIGS_NEAREST_SIGMA)
        return SPARSE_ERR_BADARG;
    if (o->backend != SPARSE_EIGS_BACKEND_AUTO && o->backend != SPARSE_EIGS_BACKEND_LANCZOS &&
        o->backend != SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART &&
        o->backend != SPARSE_EIGS_BACKEND_LOBPCG)
        return SPARSE_ERR_BADARG;
    if (o->tol < 0.0 || o->max_iterations < 0)
        return SPARSE_ERR_BADARG;
    if (o->block_size < 0 || o->block_size > n)
        return SPARSE_ERR_BADARG;
    if (o->block_size > 0 && o->block_size < k)
        return SPARSE_ERR_BADARG;
    if (o->precond_ctx && !o->precond)
        return SPARSE_ERR_BADARG;
    if (o->refine && !o->compute_vectors)
        return SPARSE_ERR_BADARG;
    if (o->refine_max_iters < 0)
        return SPARSE_ERR_BADARG;

    idx_t max_iters = 0;
    sparse_err_t err = s49_eigs_effective_max_iters(n, k, o, &max_iters);
    if (err != SPARSE_OK)
        return err;

    sparse_eigs_workspace_t *workspace = NULL;
    err = s49_eigs_handle_ensure(handle, &workspace);
    if (err != SPARSE_OK)
        return err;
    return s49_eigs_handle_prepare_backend(workspace, n, k, o, max_iters);
}

static sparse_err_t s46_validate_public_entry(const SparseMatrix *A, idx_t k,
                                              const sparse_eigs_opts_t *o,
                                              const sparse_eigs_t *result) {
    if (!A || !result)
        return SPARSE_ERR_NULL;

    idx_t n = sparse_rows(A);
    if (n != sparse_cols(A))
        return SPARSE_ERR_SHAPE;
    if (k < 1 || k > n)
        return SPARSE_ERR_BADARG;
    if (!result->eigenvalues)
        return SPARSE_ERR_NULL;
    if (o->compute_vectors && !result->eigenvectors)
        return SPARSE_ERR_NULL;
    if (o->which != SPARSE_EIGS_LARGEST && o->which != SPARSE_EIGS_SMALLEST &&
        o->which != SPARSE_EIGS_NEAREST_SIGMA)
        return SPARSE_ERR_BADARG;
    if (o->backend != SPARSE_EIGS_BACKEND_AUTO && o->backend != SPARSE_EIGS_BACKEND_LANCZOS &&
        o->backend != SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART &&
        o->backend != SPARSE_EIGS_BACKEND_LOBPCG)
        return SPARSE_ERR_BADARG;
    if (o->tol < 0.0 || o->max_iterations < 0)
        return SPARSE_ERR_BADARG;
    if (o->block_size < 0 || o->block_size > n)
        return SPARSE_ERR_BADARG;
    if (o->block_size > 0 && o->block_size < k)
        return SPARSE_ERR_BADARG;
    if (o->precond_ctx && !o->precond)
        return SPARSE_ERR_BADARG;
    if (o->refine && !o->compute_vectors)
        return SPARSE_ERR_BADARG;
    if (o->refine_max_iters < 0)
        return SPARSE_ERR_BADARG;
    if (!sparse_is_symmetric(A, 1e-12))
        return SPARSE_ERR_NOT_SPD;
    return SPARSE_OK;
}

static void s46_init_public_result(sparse_eigs_t *result, idx_t k) {
    result->n_requested = k;
    result->n_converged = 0;
    result->iterations = 0;
    result->residual_norm = 0.0;
    result->used_csc_path_ldlt = 0;
    result->peak_basis_size = 0;
    result->backend_used = SPARSE_EIGS_BACKEND_LANCZOS;
}

static sparse_eigs_backend_t s46_select_backend(idx_t n, idx_t k, const sparse_eigs_opts_t *o) {
    int explicit_lobpcg = (o->backend == SPARSE_EIGS_BACKEND_LOBPCG);
    idx_t bs_for_auto = (o->block_size > 0) ? o->block_size : k;
    int auto_lobpcg = (o->backend == SPARSE_EIGS_BACKEND_AUTO) && (o->precond != NULL) &&
                      (n >= (idx_t)SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD) && (bs_for_auto >= 4);
    if (explicit_lobpcg || auto_lobpcg)
        return SPARSE_EIGS_BACKEND_LOBPCG;

    if ((o->backend == SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART) ||
        (o->backend == SPARSE_EIGS_BACKEND_AUTO && n >= (idx_t)SPARSE_EIGS_THICK_RESTART_THRESHOLD))
        return SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART;

    return SPARSE_EIGS_BACKEND_LANCZOS;
}

static sparse_err_t s46_run_growm_backend(lanczos_op_fn op_fn, const void *op_ctx, idx_t n, idx_t k,
                                          const sparse_eigs_opts_t *o, double eff_tol,
                                          idx_t max_iters, sparse_eigs_t *result,
                                          sparse_eigs_workspace_t *workspace) {
    /* Use a single grow-m-on-retry strategy rather than alternating
     * short/long stability probes.  Reusing the same deterministic
     * v0 means each retry strictly contains the previous one: the
     * first `m_prev` Lanczos steps are reproduced exactly and only
     * the extra `m_new - m_prev` steps change. */

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
    sparse_eigs_workspace_t local_ws;
    sparse_eigs_workspace_t *growm_ws = workspace ? workspace : &local_ws;
    sparse_eigs_growm_workspace_view_t growm_view;
    sparse_err_t rc = SPARSE_ERR_NOT_CONVERGED;
    if (!workspace)
        sparse_eigs_workspace_init(growm_ws);

    /* Allocate workspace for the upper-bound Lanczos size so the
     * grow-on-retry path never reallocates.  Y_cap is m_cap × m_cap
     * (eigenvectors of T) — quadratic in m_cap but fine for the
     * practical m_cap we land on.  The multi-factor sizes (V, Y_long)
     * are validated with `sparse_size_mul_overflow` so a pathological
     * (n, m_cap) pair on a 32-bit size_t target fails cleanly with
     * SPARSE_ERR_ALLOC rather than undersizing a buffer; calloc()
     * handles its own nmemb*size overflow internally. */
    sparse_err_t alloc_err =
        sparse_eigs_workspace_prepare_growm(growm_ws, n, m_cap, k, &growm_view);
    if (alloc_err != SPARSE_OK) {
        rc = alloc_err;
        goto cleanup;
    }

    double *V = growm_view.V;
    double *alpha = growm_view.alpha;
    double *beta = growm_view.beta;
    double *v0 = growm_view.v0;
    double *theta_long = growm_view.theta_long;
    double *subdiag = growm_view.subdiag;
    double *Y_long = growm_view.Y_long;
    idx_t *sel_idx = growm_view.sel_idx;

    s20_lanczos_starting_vector(v0, n);

    /* Grow-m holds V at m_cap columns from
     * allocation until cleanup — peak basis size is `m_cap`
     * regardless of how many grow-on-retry passes actually run. */
    result->peak_basis_size = m_cap;

    idx_t total_iters = 0;
    idx_t m = m_init;
    idx_t last_m_actual = 0;
    double last_partial_res = 0.0;
    double lanczos_phase_start_s = o->progress_cb ? s29_eigs_now_s() : 0.0;

    for (;;) {
        /* Publish progress at each grow-m retry boundary.
         * Cancellation propagates through the existing cleanup
         * label, freeing all workspaces. */
        if (o->progress_cb) {
            sparse_progress_t pp = {
                .phase = "lanczos",
                .step = total_iters,
                .total = max_iters,
                .elapsed_s = s29_eigs_now_s() - lanczos_phase_start_s,
            };
            if (o->progress_cb(&pp, o->progress_user) != 0) {
                rc = SPARSE_ERR_CANCELLED;
                goto cleanup;
            }
        }
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
            /* Guard the exact-zero spectrum case as well as the
             * near-zero one.  Without the explicit `anchor == 0.0`
             * branch, `scale == 0.0 && tv_l == 0.0` would leave
             * anchor at zero and produce a 0/0 relative residual. */
            if (anchor < scale * 1e-12 || anchor == 0.0)
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
                /* For shift-invert, the Ritz values of
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
    if (!workspace)
        sparse_eigs_workspace_free(growm_ws);
    return rc;
}

static sparse_err_t s46_run_backend(sparse_eigs_backend_t backend, lanczos_op_fn op_fn,
                                    const void *op_ctx, idx_t n, idx_t k,
                                    const sparse_eigs_opts_t *o, double eff_tol, idx_t max_iters,
                                    sparse_eigs_t *result, sparse_eigs_workspace_t *workspace) {
    result->backend_used = backend;
    switch (backend) {
    case SPARSE_EIGS_BACKEND_LOBPCG:
        return s21_lobpcg_solve(op_fn, op_ctx, n, k, o, eff_tol, max_iters, result, workspace);
    case SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART:
        return s21_thick_restart_outer_loop(op_fn, op_ctx, n, k, o, eff_tol, max_iters, workspace,
                                            result);
    case SPARSE_EIGS_BACKEND_LANCZOS:
    case SPARSE_EIGS_BACKEND_AUTO:
        return s46_run_growm_backend(op_fn, op_ctx, n, k, o, eff_tol, max_iters, result, workspace);
    }

    return SPARSE_ERR_BADARG;
}

static sparse_err_t s46_sparse_eigs_sym_impl(const SparseMatrix *A, idx_t k,
                                             const sparse_eigs_opts_t *opts, sparse_eigs_t *result,
                                             sparse_eigs_workspace_t *workspace) {
    /* Library defaults when opts == NULL match the doxygen
     * contract in sparse_eigs.h. */
    const sparse_eigs_opts_t defaults = s46_default_public_opts();
    const sparse_eigs_opts_t *o = opts ? opts : &defaults;
    sparse_err_t entry_err = s46_validate_public_entry(A, k, o, result);
    if (entry_err != SPARSE_OK)
        return entry_err;

    idx_t n = sparse_rows(A);
    s46_init_public_result(result, k);

    /* Shift-invert setup for NEAREST_SIGMA.  Factor `A - sigma*I`
     * via `sparse_ldlt_factor_opts` and swap the Lanczos operator
     * from `sparse_matvec(A)` to `sparse_ldlt_solve(ldlt)`.  The
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

    /* AUTO + explicit-opt-in dispatch decision tree.
     *
     * Priority order:
     *   1. Explicit LOBPCG request.
     *   2. Explicit thick-restart Lanczos request.
     *   3. AUTO with preconditioner, large n, and adequate block size:
     *      route to LOBPCG.
     *   4. AUTO with `n >= SPARSE_EIGS_THICK_RESTART_THRESHOLD`:
     *      route to thick-restart Lanczos.
     *   5. Otherwise: grow-m Lanczos.
     *
     * The AUTO LOBPCG route requires `block_size >= 4` (defaulting
     * to `k` when `block_size == 0`) so the block work can amortise
     * the denser per-iteration costs. */
    sparse_eigs_backend_t backend = s46_select_backend(n, k, o);
    sparse_err_t rc =
        s46_run_backend(backend, op_fn, op_ctx, n, k, o, eff_tol, max_iters, result, workspace);
    sparse_ldlt_free(&ldlt_shift);
    sparse_free(A_shifted);
    return s29_maybe_refine(A, o, result, rc);
}

sparse_err_t sparse_eigs_sym(const SparseMatrix *A, idx_t k, const sparse_eigs_opts_t *opts,
                             sparse_eigs_t *result) {
    sparse_eigs_handle_t handle = {0};
    sparse_err_t err = sparse_eigs_sym_with_handle(A, k, opts, result, &handle);
    sparse_eigs_handle_free(&handle);
    return err;
}

sparse_err_t sparse_eigs_sym_with_handle(const SparseMatrix *A, idx_t k,
                                         const sparse_eigs_opts_t *opts, sparse_eigs_t *result,
                                         sparse_eigs_handle_t *handle) {
    sparse_eigs_workspace_t *workspace = NULL;
    sparse_err_t err = s49_eigs_handle_ensure(handle, &workspace);
    if (err != SPARSE_OK)
        return err;
    return s46_sparse_eigs_sym_impl(A, k, opts, result, workspace);
}

sparse_err_t sparse_eigs_sym_with_workspace_internal(const SparseMatrix *A, idx_t k,
                                                     const sparse_eigs_opts_t *opts,
                                                     sparse_eigs_t *result,
                                                     sparse_eigs_workspace_t *workspace) {
    return s46_sparse_eigs_sym_impl(A, k, opts, result, workspace);
}

/* ─── Dense Symmetric Eigensolver (Jacobi) ───────────────────────── */

/* Classical Jacobi sweeps on a dense symmetric K × K matrix.
 * Returns ascending eigenvalues in `theta_out[0..K-1]` and the
 * corresponding orthonormal eigenvectors as columns of `Q_out`
 * (K × K, column-major).  Used for arrowhead Ritz extraction
 * because the arrowhead does not have the tridiagonal shape
 * `tridiag_qr_eigenpairs` expects.
 *
 * Cost: O(K^3) per sweep × O(log K) sweeps.  For K ≤ 100 this is
 * microsecond-scale; acceptable as long as m_restart stays bounded.
 *
 * Input `A_scratch` is destroyed (overwritten with the diagonalised
 * form as a side effect). */
sparse_err_t s21_dense_sym_jacobi(double *A_scratch, idx_t K, double *theta_out, double *Q_out) {
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

/* ═══════════════════════════════════════════════════════════════════════
 * LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient
 *          (Knyazev 2001)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * LOBPCG complements the Lanczos-family backends in two regimes:
 * ill-conditioned SPD problems where a preconditioner materially
 * improves convergence, and clustered spectra where block iteration
 * can converge several target pairs together.
 *
 * Each iteration builds a Rayleigh-Ritz basis from the current
 * approximations X, the preconditioned residual block W, and the
 * previous search directions P.  The dense eigensolve over that
 * basis yields the next X/P update and the current Ritz values.
 *
 * The implementation preserves a few important numerical guards:
 *   - orthonormalise each block before forming the dense Gram matrix;
 *   - eject columns whose norm collapses below the scale-aware
 *     breakdown threshold;
 *   - keep the more stable BLOPEX-style P update for near-singular
 *     Gram matrices;
 *   - report convergence with the same relative-residual convention
 *     used by the Lanczos-family backends.
 *
 * The concrete LOBPCG implementation lives in
 * `src/sparse_eigs_lobpcg.c`, leaving this file focused on public
 * orchestration plus the shared helpers that all eigensolver
 * backends reuse.
 */

/* Orthonormalise an n × block_size_in column-major block via
 * per-column modified Gram-Schmidt with scale-aware breakdown
 * ejection.  Walks columns left-to-right, applying MGS against the
 * already-accepted columns 0..accepted-1 and either accepting (post-
 * MGS norm above threshold → normalise + advance) or ejecting
 * (post-MGS norm collapsed → linear-dependence on prior columns;
 * skip and forward-compact subsequent columns into this slot).
 *
 * Breakdown threshold uses a relative `scale * 1e-14` threshold
 * where `scale` is the running
 * max input column norm (pre-MGS), with a `DBL_MIN * 100` absolute
 * floor for the all-zero edge case. */
/* The explicit LOBPCG backend implementation lives in
 * `src/sparse_eigs_lobpcg.c`, leaving this file focused on public
 * orchestration plus the Lanczos-family backends. */
