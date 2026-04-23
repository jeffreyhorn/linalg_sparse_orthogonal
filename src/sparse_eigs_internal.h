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
 * a fixed 1e-14 threshold.  That indicates the Krylov subspace
 * span(v0, A·v0, ..., A^k·v0) has become A-invariant — T's
 * spectrum is already a subset of A's spectrum and continuing
 * the recurrence would divide by zero.
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

#endif /* SPARSE_EIGS_INTERNAL_H */
