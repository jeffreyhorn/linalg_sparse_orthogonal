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

#include "sparse_eigs.h" /* sparse_eigs_opts_t / sparse_eigs_t for LOBPCG entry point. */
#include "sparse_matrix.h"
#include "sparse_types.h"

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
 * Two consumers handle this arrowhead in the shipped code:
 *
 *   - `s21_arrowhead_to_tridiag` materialises the K × K dense
 *     symmetric form and runs classical Householder
 *     tridiagonalisation to produce `diag_out` / `subdiag_out`
 *     that the existing Sprint 20 `tridiag_qr_eigenpairs` consumes
 *     unchanged.  This path is spectrum-only (no Q returned) and is
 *     used by `tests/test_eigs_thick_restart.c` as a spectrum-
 *     preservation oracle.
 *
 *   - `s21_thick_restart_outer_loop` (the production path) runs
 *     dense Jacobi (`s21_dense_sym_jacobi`) directly on the K × K
 *     arrowhead, getting eigenvalues *and* the orthonormal Q in a
 *     single sweep — bypassing the reduction-then-extract two-step
 *     and avoiding a separate Q-accumulator pass through the
 *     Householder reduction.
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
    idx_t n;               /**< Eigenproblem vector length. */
    idx_t k_locked;        /**< Number of locked Ritz pairs carried into the next phase. */
    idx_t k_locked_cap;    /**< Allocated capacity for V_locked / theta_locked / beta_coupling. */
    double *V_locked;      /**< n × k_locked_cap, column-major; NULL when state is empty. */
    double *theta_locked;  /**< Length k_locked_cap; locked Ritz values. */
    double *beta_coupling; /**< Length k_locked_cap; trailing β_m · y_{m-1,j} entries. */
    double *residual;      /**< Length n; seeds next phase as v_{k_locked+1}.  NULL when empty. */
    double residual_norm;  /**< ||residual||.  Zero = invariant subspace reached. */
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
 * @brief Reduce a symmetric arrowhead matrix to tridiagonal form
 *        (spectrum-only — does not return the similarity Q).
 *
 * The arrowhead T produced by a thick-restart carries the locked
 * Ritz values on its top-left diagonal plus a spoke of coupling
 * entries at row/col `k_locked` (see the design block at the top of
 * `src/sparse_eigs.c`).  This helper materialises that arrowhead
 * dense and applies an orthogonal similarity to land in symmetric
 * tridiagonal form, writing `diag_out` and `subdiag_out` as the
 * reduced tridiagonal of the same spectrum.
 *
 * Algorithm.  The arrowhead is materialised as a dense K × K symmetric
 * matrix (K = k_locked + m_ext) in a scratch buffer and then
 * tridiagonalised via classical Householder reflections on the
 * symmetric matrix (Golub & Van Loan §8.3.1).  Each Householder
 * step j zeros the entries below the subdiagonal of column j via a
 * similarity H_j · T · H_j (which is also H_j · T · H_j^T because
 * H_j is symmetric).  After K - 2 steps the matrix is tridiagonal.
 * Total work is O(K^3) but K is small (typically k_locked + m_restart
 * ≤ 100), so each reduction is microsecond-scale.
 *
 * Limitation / role.  The accumulated similarity Q is *not* returned,
 * so callers cannot lift eigenvectors of the reduced tridiagonal back
 * into the arrowhead's basis through this helper alone.  The
 * production thick-restart path (`s21_thick_restart_outer_loop`) does
 * not use this helper at all — it runs dense Jacobi
 * (`s21_dense_sym_jacobi`) directly on the K × K arrowhead, which
 * yields both eigenvalues *and* the orthonormal Q in a single sweep
 * (avoids the separate Q accumulator the Day 2 design originally
 * envisioned).  This helper remains live as a spectrum-preservation
 * oracle for `test_eigs_thick_restart.c` — useful when validating
 * that future arrowhead constructions still produce the expected
 * Lanczos invariants.
 *
 * @param theta_locked    Length k_locked; locked Ritz values on the
 *                        top-left diagonal of the arrowhead.
 * @param beta_coupling   Length k_locked; the spoke entries at
 *                        position (k_locked, j) for j = 0..k_locked-1.
 *                        The last entry `beta_coupling[k_locked-1]`
 *                        lands at the standard subdiagonal position
 *                        between the locked block and the first
 *                        extension row.
 * @param k_locked        Locked subspace size (>= 1).  When
 *                        `k_locked == 1` the matrix is already
 *                        tridiagonal and the reduction is a straight
 *                        copy.
 * @param alpha_ext       Length m_ext; Lanczos alpha values for the
 *                        post-restart tridiagonal tail.  May be NULL
 *                        when m_ext == 0.
 * @param beta_ext        Length m_ext - 1; Lanczos beta values for
 *                        the tail.  May be NULL when m_ext <= 1.
 * @param m_ext           Number of post-restart Lanczos steps
 *                        (>= 0).  Let `K = k_locked + m_ext` be
 *                        the order of the arrowhead / reduced
 *                        tridiagonal.  When `m_ext == 0`,
 *                        `K == k_locked`, so there is no
 *                        extension row or tail and the arrowhead
 *                        is just the locked diagonal block.
 * @param diag_out        Length K = k_locked + m_ext; the diagonal
 *                        of the reduced tridiagonal on return.
 * @param subdiag_out     Length K - 1; the subdiagonal of the
 *                        reduced tridiagonal on return.  (Only
 *                        required when K >= 2; may be NULL when
 *                        K == 1.)
 *
 * @return SPARSE_OK on success, SPARSE_ERR_NULL / _BADARG for the
 *         usual preconditions, or SPARSE_ERR_ALLOC for the scratch
 *         buffer.
 */
sparse_err_t s21_arrowhead_to_tridiag(const double *theta_locked, const double *beta_coupling,
                                      idx_t k_locked, const double *alpha_ext,
                                      const double *beta_ext, idx_t m_ext, double *diag_out,
                                      double *subdiag_out);

/**
 * @brief Pick the locked Ritz block from a completed Lanczos phase.
 *
 * Day 2 Task 2.  Given the completed Lanczos basis V (n × m), the
 * tridiag eigenvectors Y (m × m, column-major), the eigenvalues
 * theta (length m, ascending from `tridiag_qr_eigenpairs`), and a
 * selection `sel_idx[0..take-1]` (from Sprint 20's
 * `s20_select_indices`), form the three Day 1 arrowhead state
 * pieces:
 *
 *   V_locked[:, j] = V · Y[:, sel_idx[j]]              (n × take)
 *   theta_locked[j] = theta[sel_idx[j]]                (take)
 *   beta_coupling[j] = beta_m * Y[m-1, sel_idx[j]]     (take)
 *
 * where `beta_m` is `beta[m_actual - 1]` from the Lanczos run — the
 * trailing residual norm that drives the Wu/Simon per-pair
 * residual gate.  Reuses the Sprint 20 `s20_lift_ritz_vectors`
 * kernel shape (column-major gemm inlined).
 *
 * Output buffers are caller-allocated (`V_locked_out` is
 * `n * take` doubles; the two scalar arrays are `take` doubles).
 *
 * Precondition: V's columns must be orthonormal (i.e. the Lanczos
 * run that produced V had `reorthogonalize = 1`).  The Sprint 20
 * Day 14 reliability caveat on `sparse_eigs_opts_t::reorthogonalize`
 * applies here too — without reorth, `V_locked` won't be
 * orthonormal and the arrowhead reduction's guarantees break.
 */
void lanczos_restart_pick_locked(const double *V, idx_t n, idx_t m, const double *Y,
                                 const double *theta, const idx_t *sel_idx, idx_t take,
                                 double beta_m, double *V_locked_out, double *theta_locked_out,
                                 double *beta_coupling_out);

/**
 * @brief Pack a freshly-picked locked block + residual into a
 *        `lanczos_restart_state_t`, allocating or reusing buffers
 *        as needed.
 *
 * Day 2 Task 3.  Resizes the state's `_cap` fields if the incoming
 * `k_locked` exceeds the current capacity, otherwise reuses the
 * existing buffers.  Copies the three locked-block arrays and the
 * residual vector into state-owned memory.  The state's `n` field
 * is set (or validated unchanged) from the argument.
 *
 * On failure, leaves the state in a consistent (possibly empty)
 * form and returns SPARSE_ERR_ALLOC; callers treat failure as "no
 * restart possible, emit partial results."
 *
 * @param state              Target state (populated on return).
 * @param n                  Vector length; must match `state->n`
 *                           when state is already non-empty.
 * @param k_locked           Locked subspace size.
 * @param V_locked_src       n × k_locked, column-major.
 * @param theta_locked_src   Length k_locked.
 * @param beta_coupling_src  Length k_locked.
 * @param residual_src       Length n; residual vector
 *                           `beta_m * v_{m+1}` from the Lanczos
 *                           run.
 * @param residual_norm      ||residual_src||.
 *
 * @return SPARSE_OK on success, SPARSE_ERR_NULL / _BADARG on bad
 *         args, SPARSE_ERR_ALLOC on allocation failure.
 */
sparse_err_t lanczos_restart_state_assemble(lanczos_restart_state_t *state, idx_t n, idx_t k_locked,
                                            const double *V_locked_src,
                                            const double *theta_locked_src,
                                            const double *beta_coupling_src,
                                            const double *residual_src, double residual_norm);

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
 *                  values.  The full coupling spoke
 *                  `β_coupling[0..k_locked-1]` is preserved in
 *                  `state->beta_coupling` (so the next phase / the
 *                  arrowhead reduction can reconstruct the
 *                  non-tridiagonal off-block structure); only
 *                  `beta[k_locked - 1]` is mirrored into `beta[]`
 *                  so the active-block layout stays tridiagonal.
 * @param m_actual  Output count of Lanczos columns produced in V
 *                  (includes the k_locked prefix).
 *
 * @return SPARSE_OK on success, SPARSE_ERR_NULL / _SHAPE / _BADARG
 *         / _ALLOC for the usual preconditions, or any error the
 *         operator callback propagates.
 *
 * Executes one thick-restart Lanczos phase against `op`, starting
 * either from a fresh `v0` when `state` is empty or from the
 * locked / residual information carried in `state` after a prior
 * restart.  On return, `V` / `alpha` / `beta` describe the current
 * phase's basis and tridiagonal data, while the preserved locked
 * subspace, coupling terms, and residual needed for the next
 * restart remain represented via `state`.
 */
sparse_err_t lanczos_thick_restart_iterate(lanczos_op_fn op, const void *ctx, idx_t n,
                                           const double *v0, idx_t m_restart, int reorthogonalize,
                                           lanczos_restart_state_t *state, double *V, double *alpha,
                                           double *beta, idx_t *m_actual);

/* ═══════════════════════════════════════════════════════════════════════
 * LOBPCG building blocks (Knyazev 2001)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * The LOBPCG backend reuses the `lanczos_op_fn` callback type for
 * `A · X` multiplies (one application per column of the
 * n × block_size X / W / P blocks; a future block-matvec is a
 * potential follow-up) and the `sparse_precond_fn` callback from
 * `sparse_iterative.h` for the preconditioner that turns the
 * residual block R into the preconditioned-residual block
 * W = M^{-1} · R.
 *
 * The full pipeline — block Rayleigh-Ritz over `[X | W | P]`,
 * BLOPEX-style conditioning guard, soft-locking, and AUTO
 * dispatch — is documented in the LOBPCG design block at the top
 * of `src/sparse_eigs.c` (between the Lanczos and thick-restart
 * blocks). */

/**
 * @brief Modified Gram-Schmidt orthonormalisation of an n × block_size
 *        column-major block.
 *
 * Walks the columns of `Q` left-to-right.  For each column j:
 *   - Apply MGS against the previously-orthonormalised columns
 *     0..j-1 of Q.
 *   - Compute the resulting column norm.
 *   - If the norm exceeds the breakdown threshold (`scale * 1e-14`,
 *     where `scale` is the running max of accepted column norms with
 *     a `DBL_MIN * 100` floor), normalise the column and accept it.
 *   - Otherwise eject the column (overwrite with the next accepted
 *     column from the right and decrement the trailing pointer); the
 *     effective block size shrinks accordingly.
 *
 * Mirrors the `s21_mgs_reorth` kernel applied per-column with the
 * breakdown convention from commit 70015a4.  Preserves numerical
 * stability of the downstream Rayleigh-Ritz step, which requires
 * the basis `[X, W, P]` to be orthonormal so the Gram matrix
 * `Q^T · A · Q` is symmetric (off by O(eps · cond) terms otherwise).
 *
 * @param Q              n × block_size_in, column-major.  Modified
 *                       in place: accepted columns are normalised
 *                       and orthogonal to all earlier accepted
 *                       columns; ejected columns are removed by
 *                       compacting from the right, so on return only
 *                       the first *block_size_out columns are
 *                       meaningful.
 * @param n              Vector length.
 * @param block_size_in  Input block size (>= 0; 0 is a no-op).
 * @param block_size_out Output: number of accepted (non-ejected)
 *                       columns.  0 <= *block_size_out <= block_size_in.
 *
 * @return SPARSE_OK on success, SPARSE_ERR_NULL / SPARSE_ERR_BADARG
 *         on invalid arguments.
 */
sparse_err_t s21_lobpcg_orthonormalize_block(double *Q, idx_t n, idx_t block_size_in,
                                             idx_t *block_size_out);

/**
 * @brief One block Rayleigh-Ritz step over the LOBPCG subspace
 *        `[X, W, P]`.
 *
 * Concatenates the three n × block_size matrices into an
 * n × (3·block_size) basis Q, orthonormalises it (via
 * `s21_lobpcg_orthonormalize_block`), forms the Gram matrix
 * `G = Q^T · A · Q` of size 3·block_size × 3·block_size (symmetric)
 * by applying `op` columnwise, runs the dense symmetric Jacobi
 * eigensolver `s21_dense_sym_jacobi` to extract the Ritz pairs of
 * `G`, selects the `block_size` Ritz pairs that match `which`, and
 * forms the next X / P from the combination coefficients
 * (Knyazev 2001 eq. 2.11; the BLOPEX-style robust formulation
 * lives in `s21_lobpcg_solve` as a conditioning guard, see
 * Sprint 21 Day 9 retrospective).
 *
 * @param op             Symmetric linear operator (A or shift-invert).
 * @param ctx            Opaque context for `op`.
 * @param n              Vector length.
 * @param block_size     LOBPCG block size.  X/W/P are each n × block_size.
 * @param X              In/out: current eigenvector estimates,
 *                       n × block_size, column-major.  Replaced with
 *                       the next iterate on return.
 * @param W              In/out: preconditioned residual block,
 *                       n × block_size, column-major.  Used as input
 *                       only; caller refills for the next iteration.
 * @param P              In/out: previous search direction block,
 *                       n × block_size, column-major.  Replaced with
 *                       the new direction on return.  May be NULL on
 *                       the first iteration (signals "no P yet").
 * @param which          Spectrum-selection mode (LARGEST / SMALLEST /
 *                       NEAREST_SIGMA via the same op-negation /
 *                       shift-invert wrappers Lanczos uses).
 * @param theta_out      Output: block_size Ritz values; ordered to
 *                       match `which`.
 *
 * @return SPARSE_OK on success, or any error from `op` /
 *         allocation / dense eigensolve.
 */
sparse_err_t s21_lobpcg_rr_step(lanczos_op_fn op, const void *ctx, idx_t n, idx_t block_size,
                                double *X, double *W, double *P, sparse_eigs_which_t which,
                                double *theta_out);

/**
 * @brief LOBPCG outer loop — full block iteration to convergence.
 *
 * Initialises X to a deterministic block (Sprint 20 golden-ratio
 * fractional mixing extended across `block_size` columns) and
 * P := 0.  Each iteration:
 *
 *   1. Compute residual `R = A·X − X·diag(theta)` for the current
 *      Ritz values theta.
 *   2. Convergence gate per column j: if
 *      `||R[:, j]|| / max(|theta_j|, scale) <= tol`, mark column j
 *      as converged.  When all `k` selected columns are converged,
 *      emit the result and return SPARSE_OK.
 *   3. Apply the preconditioner: `W[:, j] = precond(R[:, j])` when
 *      `precond != NULL`, else `W[:, j] = R[:, j]` (vanilla LOBPCG).
 *   4. Run `s21_lobpcg_rr_step(X, W, P)` — updates X, P, theta in
 *      place.
 *
 * On reaching `max_iters` without converging all `k` columns,
 * returns SPARSE_ERR_NOT_CONVERGED with the current best estimates
 * filled in (matches the Sprint 20 grow-m partial-results contract).
 *
 * @param op             Symmetric linear operator.
 * @param ctx            Opaque context for `op`.
 * @param n              Vector length.
 * @param k              Number of converged eigenpairs requested.
 * @param o              Caller's options struct (provides
 *                       `block_size`, `precond`, `precond_ctx`,
 *                       `which`, `tol`, `compute_vectors`, `sigma`).
 * @param eff_tol        Effective convergence tolerance (already
 *                       resolved in `sparse_eigs_sym`).
 * @param max_iters      Effective iteration cap (already resolved).
 * @param result         Output struct; `eigenvalues` /
 *                       `eigenvectors` buffers caller-allocated.
 *
 * @return SPARSE_OK if all k pairs converged.
 * @return SPARSE_ERR_NOT_CONVERGED on iteration-cap exhaustion with
 *         partial results filled in.
 * @return SPARSE_ERR_NULL / _BADARG / _ALLOC for the usual
 *         preconditions, or any error from `op` / `precond` /
 *         allocation.
 */
sparse_err_t s21_lobpcg_solve(lanczos_op_fn op, const void *ctx, idx_t n, idx_t k,
                              const sparse_eigs_opts_t *o, double eff_tol, idx_t max_iters,
                              sparse_eigs_t *result);

#endif /* SPARSE_EIGS_INTERNAL_H */
