#include "sparse_alloc_internal.h"
#include "sparse_eigs_internal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Thick-restart backend extracted from `src/sparse_eigs.c` so the
 * main eigensolver file can stay focused on shared orchestration and
 * the non-thick-restart backends. */

/* ═══════════════════════════════════════════════════════════════════════
 * Thick-restart Lanczos (Wu/Simon arrowhead)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The grow-m Lanczos outer loop converges reliably on the target
 * corpus but holds the full Lanczos basis `V` across all retries —
 * peak memory `O(m_cap · n)`.  Thick-restart preserves only the
 * converged Ritz subspace across restarts and re-runs Lanczos from
 * a small locked-plus-residual basis.  Peak memory drops to
 * `O((k_locked + m_restart) · n)`; at k = 5, m_restart = 30 this
 * is roughly 35 columns regardless of total iteration count.
 *
 * Restart protocol.
 *
 *   1. Run a Lanczos phase of length `m_restart`, writing α / β / V
 *      as usual.  `lanczos_thick_restart_iterate` with an empty
 *      state is equivalent to `lanczos_iterate_op`.
 *   2. On non-convergence, extract Ritz pairs (θ_j, y_j) from T via
 *      `s20_ritz_pairs` and select the top k (or matching `which`)
 *      via `s20_select_indices`.
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
 * The spectrum-only arrowhead reduction helper converts this shape
 * back to a symmetric tridiagonal when needed.
 *
 * Why keep the grow-m path.  Grow-m is simpler and its constants
 * are good on small-to-moderate n.  The thick-restart path is only
 * a clear win when the basis would otherwise grow too large, which
 * is why AUTO dispatch uses `SPARSE_EIGS_THICK_RESTART_THRESHOLD`.
 *
 * Field ownership.  `lanczos_restart_state_t` owns its allocations
 * (V_locked / theta_locked / beta_coupling / residual) once
 * populated; `lanczos_restart_state_free` releases them.  An
 * empty state (zeroed struct) is legal input and represents a
 * fresh start.  The assembly helpers allocate on first use sized to
 * `k_locked_cap` and reuse the buffers across subsequent restarts
 * when `k_locked <= k_locked_cap` holds.
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

/* ─── Arrowhead reduction + Ritz locking helpers ─────────────────── */

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
 *   - Spectrum-only reduction.  The production path uses dense
 *     Jacobi when it needs both eigenvalues and the orthogonal
 *     transform.
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

    /* Dense K×K scratch with checked count arithmetic. */
    size_t K_size = 0, K2 = 0;
    if (sparse_idx_to_size_checked(K, &K_size) || sparse_size_mul_overflow(K_size, K_size, &K2))
        return SPARSE_ERR_ALLOC;
    double *T = NULL;
    if (sparse_calloc_array(K2, sizeof(double), (void **)&T) != SPARSE_OK) {
        return SPARSE_ERR_ALLOC;
    }

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
        double *v = NULL;
        double *p = NULL;
        double *q = NULL;
        if (sparse_calloc_idx_array(K, sizeof(double), (void **)&v) != SPARSE_OK ||
            sparse_calloc_idx_array(K, sizeof(double), (void **)&p) != SPARSE_OK ||
            sparse_calloc_idx_array(K, sizeof(double), (void **)&q) != SPARSE_OK) {
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
            for (idx_t row = j + 2; row < K; row++) {
                T_AT(row, j) = 0.0;
                T_AT(j, row) = 0.0;
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

/* Assemble the three locked-block arrays from a completed Lanczos
 * phase's Ritz pairs.
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

/* Pack the picked locked block + residual into `state`, allocating
 * buffers if the current capacity is insufficient.  Reuses existing
 * buffers when `k_locked <= state->k_locked_cap` and `state->n == n`. */
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

    size_t n_size = 0, k_locked_size = 0, v_elems = 0;
    if (sparse_idx_to_size_checked(n, &n_size) ||
        sparse_idx_to_size_checked(k_locked, &k_locked_size))
        return SPARSE_ERR_ALLOC;
    if (k_locked > 0 && sparse_size_mul_overflow(n_size, k_locked_size, &v_elems))
        return SPARSE_ERR_ALLOC;

    /* Allocate or grow V_locked capacity if needed.  We keep the
     * residual buffer sized to n regardless of k_locked, so it's
     * allocated separately below. */
    if (k_locked > state->k_locked_cap) {
        double *new_V = NULL;
        double *new_theta = NULL;
        double *new_beta = NULL;
        if (sparse_malloc_array(v_elems, sizeof(double), (void **)&new_V) != SPARSE_OK ||
            sparse_malloc_idx_array(k_locked, sizeof(double), (void **)&new_theta) != SPARSE_OK ||
            sparse_malloc_idx_array(k_locked, sizeof(double), (void **)&new_beta) != SPARSE_OK) {
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
        if (sparse_malloc_idx_array(n, sizeof(double), (void **)&state->residual) != SPARSE_OK)
            return SPARSE_ERR_ALLOC;
    }

    /* Copy the locked block + residual into state-owned memory. */
    if (k_locked > 0) {
        memcpy(state->V_locked, V_locked_src, v_elems * sizeof(double));
        memcpy(state->theta_locked, theta_locked_src, k_locked_size * sizeof(double));
        memcpy(state->beta_coupling, beta_coupling_src, k_locked_size * sizeof(double));
    }
    memcpy(state->residual, residual_src, n_size * sizeof(double));

    state->n = n;
    state->k_locked = k_locked;
    state->residual_norm = residual_norm;
    return SPARSE_OK;
}

/* Run one Lanczos phase of length `m_restart` against the
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

    /* Empty-state fast path: delegate to the standard Lanczos helper. */
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
     * Mirrors the shared `lanczos_iterate_op` inner body but
     * with a starting step index of k_locked and an augmented
     * V whose first k_locked columns are the locked block.  Full-
     * MGS reorth against V[:, 0..k) at each step handles the
     * arrowhead-spoke subtraction implicitly. */
    double *w = NULL;
    if (sparse_malloc_idx_array(n, sizeof(double), (void **)&w) != SPARSE_OK)
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

/* ─── Thick-restart outer loop ───────────────────────────────────── */

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

/* Wu/Simon thick-restart dispatch.  Called from `sparse_eigs_sym`
 * when AUTO or an explicit backend request selects this path.
 *
 * Manages the phase-by-phase restart loop with bounded memory:
 * V / alpha / beta are sized to `m_restart` (fixed), not the
 * monotone-growing `m_cap` of the grow-m path.  Peak
 * memory is `O((m_restart + k_locked_cap) · n)`, independent of
 * total iteration count.
 *
 * Convergence gate mirrors the grow-m path: Wu/Simon per-pair
 * residual `|beta_last · Y_arrow[m_actual - 1, j]|` scaled by
 * `max(|theta_j|, scale)`.  When `beta_last` is the last Lanczos
 * beta of the CURRENT phase (not the prior-phase spoke), the
 * identity `||A V_aug y - θ V_aug y|| = |beta_last · y_last|` holds
 * across the augmented (locked + new) subspace by the same Paige
 * derivation the grow-m path uses.
 *
 * Arguments are the pre-processed outer-loop inputs from
 * `sparse_eigs_sym` (operator + context + shift-invert state).
 * Result is populated on exit. */
sparse_err_t s21_thick_restart_outer_loop(lanczos_op_fn op, const void *ctx, idx_t n, idx_t k,
                                          const sparse_eigs_opts_t *o, double eff_tol,
                                          idx_t max_iters, sparse_eigs_workspace_t *workspace,
                                          sparse_eigs_t *result) {
    /* Restart basis size.  `2k + 20` keeps peak
     * `V + V_locked` at ~`m_restart + k = 3k + 20` columns, which
     * for `k = 5` gives 35 columns — roughly 15× smaller than the
     * grow-m path's typical `m_cap = 500` while still leaving
     * enough of a Krylov spectrum per phase to converge extreme
     * Ritz values without letting the basis grow unbounded.
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
    sparse_eigs_workspace_t local_ws;
    sparse_eigs_workspace_t *thick_ws = workspace ? workspace : &local_ws;
    sparse_eigs_thick_restart_workspace_view_t thick_view;
    if (!workspace)
        sparse_eigs_workspace_init(thick_ws);

    /* Peak simultaneous V columns = m_restart
     * (main buffer) + k (locked state across restarts) + k (the
     * transient `V_locked_tmp` during pick_locked, briefly live
     * alongside both V and state->V_locked).  On a grow-m run
     * with m_cap = 500, k = 5 this metric lands at 510, so the
     * bcsstk14 parity test captures the memory savings by
     * comparing peak_basis_size ratios rather than absolute
     * numbers. */
    result->peak_basis_size = m_restart + 2 * k;

    lanczos_restart_state_t state = {0};

    sparse_err_t rc = SPARSE_ERR_NOT_CONVERGED;
    sparse_err_t ws_err =
        sparse_eigs_workspace_prepare_thick_restart(thick_ws, n, m_restart, k, &thick_view);
    if (ws_err != SPARSE_OK) {
        rc = ws_err;
        goto cleanup;
    }

    double *V = thick_view.V;
    double *alpha = thick_view.alpha;
    double *beta = thick_view.beta;
    double *v0 = thick_view.v0;
    double *residual_vec = thick_view.residual_vec;
    double *T_arrow = thick_view.T_arrow;
    double *theta_arrow = thick_view.theta_arrow;
    double *Y_arrow = thick_view.Y_arrow;
    idx_t *sel_idx = thick_view.sel_idx;
    double *V_locked_tmp = thick_view.V_locked_tmp;
    double *theta_locked_tmp = thick_view.theta_locked_tmp;
    double *beta_coupling_tmp = thick_view.beta_coupling_tmp;

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
         * dense Jacobi (bypasses the spectrum-only reduce-to-tridiag
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
     * partial results from the last phase, matching the grow-m
     * path's final-phase fallthrough. */
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
    if (!workspace)
        sparse_eigs_workspace_free(thick_ws);
    lanczos_restart_state_free(&state);
    return rc;
}
