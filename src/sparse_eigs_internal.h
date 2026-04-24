#ifndef SPARSE_EIGS_INTERNAL_H
#define SPARSE_EIGS_INTERNAL_H

/**
 * @file sparse_eigs_internal.h
 * @brief Internal (non-public) entry points for the symmetric
 *        eigensolver.  Used by unit tests that exercise individual
 *        Lanczos building blocks before Day 11 wires the full
 *        `sparse_eigs_sym` path.
 *
 * The public API lives in `include/sparse_eigs.h`; nothing here is
 * shipped in the library's public headers.
 */

#include "sparse_matrix.h"
#include "sparse_types.h"

#include <stddef.h>

/**
 * @brief m-step Lanczos recurrence on symmetric A, optionally
 *        with full reorthogonalization against all prior Lanczos
 *        vectors.
 *
 * See the Lanczos design block at the top of `src/sparse_eigs.c`
 * for the classical 3-term recurrence, finite-precision caveats,
 * and the reorthogonalization rationale.  Day 8 landed the no-
 * reorth recurrence; Day 9 added the `reorthogonalize` gate.
 *
 * On success, writes:
 *
 *   V[i + k * n] = k-th Lanczos vector (column-major; caller
 *                   allocates at least n * m_max doubles).
 *   alpha[k]     = T[k, k]           (length m_max doubles)
 *   beta[k]      = T[k, k+1]         (length m_max doubles; only
 *                   [0, *m_actual - 1) populates the tridiagonal
 *                   super/subdiagonal; beta[*m_actual - 1] stores
 *                   the final iteration's residual norm and is
 *                   not part of the T matrix).
 *   *m_actual    = number of Lanczos iterations executed
 *                  (<= m_max; less than m_max when an invariant
 *                  subspace is detected, i.e. beta_k ≈ 0).
 *
 * Early exit: the iteration stops as soon as beta_k falls below
 * the breakdown tolerance used by the recurrence, namely
 * max(t_norm * 1e-14, DBL_MIN * 100), where t_norm tracks the
 * current tridiagonal scale (running max of row-k row-sums
 * |beta_{k-1}| + |alpha_k| + |beta_k|).  That indicates the
 * Krylov subspace span(v0, A·v0, ..., A^k·v0) has become
 * A-invariant — T's spectrum is already a subset of A's spectrum
 * and continuing the recurrence would divide by (numerically)
 * zero.
 *
 * Reorthogonalization.  When `reorthogonalize != 0`, after each
 * step's 3-term recurrence produces the tentative w, the helper
 * subtracts the projection of w onto every prior Lanczos vector
 * (modified Gram-Schmidt, MGS).  MGS has the same asymptotic cost
 * as classical Gram-Schmidt but is numerically more stable under
 * cancellation — the standard choice for Lanczos.  The per-step
 * extra cost is O(k·n), making the overall iteration O(m²·n)
 * instead of the basic recurrence's O(m·n).  On wide-spectrum
 * matrices (condition number > 10⁶) the overhead pays for itself
 * many times over by suppressing ghost Ritz values.
 *
 * @param A        Symmetric square matrix (not modified).
 * @param v0       Starting vector, length `sparse_rows(A)`.  Must
 *                 not be the zero vector (‖v0‖ < 1e-14 rejects
 *                 as SPARSE_ERR_BADARG).
 * @param m_max    Maximum number of Lanczos iterations.  Caller's
 *                 `V` / `alpha` / `beta` buffers must be sized for
 *                 this many iterations.  1 <= m_max <= n.
 * @param reorthogonalize  If nonzero, apply MGS full
 *                 reorthogonalization against V[:, 0..k) after
 *                 each 3-term step.  If zero, the basic
 *                 recurrence only (Day 8 behaviour; ghost Ritz
 *                 values possible on wide-spectrum A).
 * @param V        Output Lanczos basis (n × m_max, column-major).
 * @param alpha    Output T diagonal, length m_max.
 * @param beta     Output T sub/super-diagonal, length m_max.
 * @param m_actual Output count of Lanczos iterations executed.
 *
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_SHAPE (A not
 *         square), SPARSE_ERR_BADARG (m_max out of range, ‖v0‖
 *         too small), SPARSE_ERR_ALLOC (workspace).
 */
sparse_err_t lanczos_iterate(const SparseMatrix *A, const double *v0, idx_t m_max,
                             int reorthogonalize, double *V, double *alpha, double *beta,
                             idx_t *m_actual);

/**
 * @brief Operator type for generalised Lanczos drivers.
 *
 * Applies `y := op(x)` for some symmetric linear operator `op`.
 * `ctx` is an opaque pointer threaded through by the caller (e.g.
 * `(const sparse_ldlt_t *)` for shift-invert mode).  `n` is the
 * vector length; redundant with ctx but kept explicit so the helper
 * can validate inputs without peeking at the ctx struct.
 *
 * Sprint 20 Day 12 adds this indirection so `lanczos_iterate_op`
 * can drive the 3-term recurrence against either `sparse_matvec(A)`
 * (default) or `(A - sigma*I)^{-1}·v` via `sparse_ldlt_solve`
 * (shift-invert).  The operator is expected to be symmetric — the
 * caller is responsible for passing only symmetric operators.
 */
typedef sparse_err_t (*lanczos_op_fn)(const void *ctx, idx_t n, const double *x, double *y);

/**
 * @brief m-step Lanczos recurrence on a symmetric operator supplied
 *        via callback.
 *
 * Same semantics as `lanczos_iterate` (3-term recurrence, optional
 * full MGS reorthogonalization, invariant-subspace early-exit) but
 * calls `op(ctx, n, v_k, w)` to compute `w = op · v_k` instead of
 * `sparse_matvec`.  Used by shift-invert Lanczos (Sprint 20 Day 12)
 * where `op` is `(A - sigma*I)^{-1}` via a pre-computed LDL^T
 * factorization.
 *
 * All output arguments follow the same conventions as
 * `lanczos_iterate`.  Errors from `op` propagate as-is (e.g. a
 * singular pivot during LDL^T solve surfaces as
 * `SPARSE_ERR_SINGULAR`).
 */
sparse_err_t lanczos_iterate_op(lanczos_op_fn op, const void *ctx, idx_t n, const double *v0,
                                idx_t m_max, int reorthogonalize, double *V, double *alpha,
                                double *beta, idx_t *m_actual);

/* ═══════════════════════════════════════════════════════════════════════
 * Sprint 21 Day 1: Thick-restart Lanczos data structures + entry point
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The Sprint 20 Day 13 growing-m outer loop grows the Krylov basis
 * up to `m_cap` across retries, with peak memory `O(m_cap · n)`.
 * For large-n SuiteSparse matrices (bcsstk14 at n = 1806) this is
 * already ~26 MB of V at `m_cap = n`.  Sprint 21 Day 1 introduces
 * a thick-restart mechanism (Wu/Simon 2000; Stathopoulos/Saad 2007)
 * that preserves the converged Ritz subspace in a compact "locked"
 * block and continues Lanczos from a trailing residual, so peak
 * memory drops to `O((k_locked + m_restart) · n)`.
 *
 * Data layout.  After a restart that locks `k_locked` Ritz pairs:
 *
 *   V_locked   : n × k_locked    — the locked Ritz vectors (V · Y_k)
 *   theta_locked : k_locked      — their Ritz values
 *   beta_coupling : k_locked     — the trailing `beta_m * y_{m-1,j}`
 *                                  coupling entries for each locked
 *                                  Ritz vector; these seed row
 *                                  (k_locked) of the arrowhead T.
 *   residual   : n               — `beta_m * v_{m+1}` at the moment
 *                                  of restart; norm is
 *                                  `residual_norm`.  Normalised and
 *                                  used as the next Lanczos vector.
 *   residual_norm : scalar       — ||residual||; equals the
 *                                  original beta_m.  Zero signals
 *                                  an invariant subspace (no more
 *                                  restarts possible).
 *
 * Arrowhead T after a restart.  Rows/columns 0..k_locked-1 hold
 * the locked θ values on the diagonal and β_coupling on the last
 * row/column of that block (zero elsewhere in the block).  Row
 * k_locked onward contains the standard Lanczos α/β entries the
 * next phase writes.  Sketch for k_locked = 3, m_restart = 5:
 *
 *       [ θ_0   0    0    β_0   0    0    0   ]
 *       [  0   θ_1   0    β_1   0    0    0   ]
 *       [  0    0   θ_2   β_2   0    0    0   ]
 *       [ β_0  β_1  β_2   α_3  β_3   0    0   ]
 *       [  0    0    0    β_3  α_4  β_4   0   ]
 *       [  0    0    0    0    β_4  α_5  β_5  ]
 *       [  0    0    0    0    0    β_5  α_6  ]
 *
 * The arrowhead-to-tridiagonal reduction via Givens rotations (Day
 * 2 task) chases the trailing coupling entries `β_0 β_1 β_2` down
 * to a symmetric tridiagonal that the existing Sprint 20
 * `tridiag_qr_eigenpairs` can consume unchanged.
 */

/**
 * @brief Restart state preserved across Lanczos thick-restart phases.
 *
 * Zero-initialised (`{0}`) represents an empty state — legal input
 * to `lanczos_thick_restart_iterate` for a fresh Lanczos start.
 * After a restart phase, the struct owns the `V_locked`,
 * `theta_locked`, `beta_coupling`, and `residual` buffers and the
 * caller must release them via `lanczos_restart_state_free`.
 *
 * Field invariants (non-empty state):
 *   - `n > 0` is the vector length matching the outer eigenproblem.
 *   - `0 <= k_locked <= k_locked_cap` is the current locked count.
 *   - `V_locked` is column-major, n × k_locked_cap allocated.
 *   - `theta_locked` / `beta_coupling` have length k_locked_cap.
 *   - `residual` has length n; `residual_norm == ||residual||`.
 *
 * The `_cap` fields carry the allocated capacity so the outer loop
 * can grow k_locked across restarts without reallocating when
 * capacity suffices.
 */
typedef struct lanczos_restart_state {
    idx_t n;             /**< Eigenproblem vector length. */
    idx_t k_locked;      /**< Number of locked Ritz pairs carried into the next phase. */
    idx_t k_locked_cap;  /**< Allocated capacity for V_locked / theta_locked / beta_coupling. */
    double *V_locked;    /**< n × k_locked_cap, column-major; NULL when state is empty. */
    double *theta_locked; /**< Length k_locked_cap; locked Ritz values. */
    double *beta_coupling; /**< Length k_locked_cap; trailing β_m · y_{m-1,j} entries. */
    double *residual;    /**< Length n; seeds next phase as v_{k_locked+1}.  NULL when empty. */
    double residual_norm; /**< ||residual||.  Zero = invariant subspace reached. */
} lanczos_restart_state_t;

/**
 * @brief Release any allocations owned by a restart state and
 *        reset it to the empty form.
 *
 * Safe to call on a zero-initialised struct (no-op).  After return,
 * the struct is legal input to another `lanczos_thick_restart_iterate`
 * call (which will treat it as a fresh start).  `state == NULL` is
 * a no-op.
 */
void lanczos_restart_state_free(lanczos_restart_state_t *state);

/**
 * @brief One phase of thick-restart Lanczos: either a fresh start
 *        (empty `state`) or a continuation from a locked subspace
 *        produced by a previous phase.
 *
 * Semantics match `lanczos_iterate_op` for the numeric recurrence,
 * plus:
 *   - If `state->k_locked == 0` or `state->V_locked == NULL` (empty
 *     state), the phase behaves exactly like `lanczos_iterate_op`:
 *     normalises `v0`, builds an m_restart-step Lanczos basis in V,
 *     writes α / β.
 *   - If `state` carries a non-empty locked subspace, the phase
 *     copies `V_locked` into V[:, 0 .. k_locked-1], seeds
 *     v_{k_locked} from `state->residual / state->residual_norm`,
 *     writes the arrowhead rows 0..k_locked-1 of α / β from
 *     `theta_locked` / `beta_coupling`, and continues the standard
 *     3-term recurrence from step k_locked onward.  The resulting
 *     V spans the same subspace as the combined locked + new
 *     Krylov contributions; T has the arrowhead shape sketched
 *     above.
 *
 * The caller is responsible for sizing V / alpha / beta to at least
 * m_restart entries (V is n × m_restart).  `*m_actual` on return is
 * the total column count written to V (including the k_locked
 * locked columns); 0 only on error.
 *
 * The `reorthogonalize` flag controls only the new Lanczos steps —
 * the locked columns are already orthonormal (by construction from
 * the previous phase's Ritz extraction) and the seeding residual
 * is orthogonalised against them as part of the restart setup.
 *
 * @param op        Symmetric linear operator callback (as
 *                  `lanczos_iterate_op`).
 * @param ctx       Opaque context for `op`.
 * @param n         Vector length.
 * @param v0        Starting vector; used only when `state` is
 *                  empty.  May be NULL when `state` carries a
 *                  non-empty locked subspace.
 * @param m_restart Target phase length (total columns written to V;
 *                  must be > state->k_locked when state is
 *                  non-empty, and <= n).
 * @param reorthogonalize  Full-MGS reorth flag.  Applies only to
 *                  the new Lanczos steps after step k_locked.
 * @param state     Optional thick-restart state.  May be NULL for a
 *                  fresh start (equivalent to passing a zeroed
 *                  struct).  On non-NULL input, `state->k_locked`
 *                  <= n must hold.
 * @param V         Output Lanczos basis (n × m_restart).
 * @param alpha     Output T diagonal (length m_restart).  Rows
 *                  0..k_locked-1 hold `theta_locked` (arrowhead
 *                  diagonal); rows k_locked.. hold standard
 *                  Lanczos α values.
 * @param beta      Output T sub/super-diagonal (length m_restart).
 *                  Rows 0..k_locked-2 hold 0 (the locked block is
 *                  diagonal except for its last row/column); row
 *                  k_locked-1 holds the coupling vector's last
 *                  entry; rows k_locked.. hold standard Lanczos β
 *                  values.  The trailing β entries `β_coupling[0..
 *                  k_locked-1]` are emitted via a separate caller-
 *                  provided buffer (to avoid reshaping `beta` into
 *                  a non-tridiagonal layout) — passed via the Day
 *                  2 arrowhead-reduction helper.
 * @param m_actual  Output count of Lanczos columns produced in V
 *                  (includes the k_locked prefix).
 *
 * @return SPARSE_OK on success, SPARSE_ERR_NULL / _SHAPE / _BADARG
 *         / _ALLOC for the usual preconditions, or any error the
 *         operator callback propagates.
 *
 * **Day 1 stub.** Returns SPARSE_ERR_BADARG ("stub in progress"
 * signal, consistent with the Sprint 20 Day 7 convention — no
 * SPARSE_ERR_NOT_IMPL exists in this codebase).  Days 2-3 replace
 * the body with the arrowhead reduction + phase execution.
 */
sparse_err_t lanczos_thick_restart_iterate(lanczos_op_fn op, const void *ctx, idx_t n,
                                           const double *v0, idx_t m_restart, int reorthogonalize,
                                           lanczos_restart_state_t *state, double *V, double *alpha,
                                           double *beta, idx_t *m_actual);

#endif /* SPARSE_EIGS_INTERNAL_H */
