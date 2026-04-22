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
 * @brief Basic 3-term Lanczos recurrence on symmetric A (no
 *        reorthogonalization).
 *
 * See the Lanczos design block at the top of `src/sparse_eigs.c`
 * for algorithm details, finite-precision caveats, and the
 * roadmap for Days 9-11 (full reorth / thick-restart / Ritz
 * extraction).
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
 * @param A        Symmetric square matrix (not modified).
 * @param v0       Starting vector, length `sparse_rows(A)`.  Must
 *                 not be the zero vector (‖v0‖ < 1e-14 rejects
 *                 as SPARSE_ERR_BADARG).
 * @param m_max    Maximum number of Lanczos iterations.  Caller's
 *                 `V` / `alpha` / `beta` buffers must be sized for
 *                 this many iterations.  1 <= m_max <= n.
 * @param V        Output Lanczos basis (n × m_max, column-major).
 * @param alpha    Output T diagonal, length m_max.
 * @param beta     Output T sub/super-diagonal, length m_max.
 * @param m_actual Output count of Lanczos iterations executed.
 *
 * @return SPARSE_OK, SPARSE_ERR_NULL, SPARSE_ERR_SHAPE (A not
 *         square), SPARSE_ERR_BADARG (m_max out of range, ‖v0‖
 *         too small), SPARSE_ERR_ALLOC (workspace).
 */
sparse_err_t lanczos_iterate_basic(const SparseMatrix *A, const double *v0, idx_t m_max, double *V,
                                   double *alpha, double *beta, idx_t *m_actual);

#endif /* SPARSE_EIGS_INTERNAL_H */
