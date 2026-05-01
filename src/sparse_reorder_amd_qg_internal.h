#ifndef SPARSE_REORDER_AMD_QG_INTERNAL_H
#define SPARSE_REORDER_AMD_QG_INTERNAL_H

/**
 * @file sparse_reorder_amd_qg_internal.h
 * @brief Quotient-graph AMD — internal helper (Amestoy/Davis/Duff 2004).
 *
 * The implementation that the public `sparse_reorder_amd`
 * (`include/sparse_reorder.h`) forwards to.  Lives behind an
 * internal header because the test suite calls it directly to
 * verify wrapper delegation.  The public AMD path goes through
 * `sparse_reorder_amd` only.
 *
 * Contract: same signature + same permutation convention as
 * `sparse_reorder_amd`, i.e. `perm[i]` is the i-th vertex
 * eliminated.  Workspace starts at `≈ 5·nnz + 6·n + 1` integer
 * entries and grows on demand; see the public header for memory
 * notes.
 */

#include "sparse_matrix.h"
#include "sparse_types.h"

/**
 * @brief Quotient-graph Approximate Minimum Degree ordering.
 *
 * Same input/output contract as `sparse_reorder_amd` from
 * `include/sparse_reorder.h`, implemented with a single integer
 * workspace (initial size `≈ 5·nnz + 6·n + 1`, grows on demand)
 * instead of an n×n bitset.
 *
 * @param A     Input matrix (must be square, not modified).
 * @param perm  Output permutation array, length n; `perm[new_i] =
 *              old_i`.  Caller pre-allocates.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or perm is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC on allocation failure (or when the
 *         requested workspace size cannot be represented safely).
 * @return SPARSE_ERR_BADARG if the elimination loop's invariant
 *         is violated (the quotient-graph state would have to be
 *         corrupted; asserts in debug builds, surfaces under
 *         NDEBUG so callers can propagate the failure cleanly).
 */
sparse_err_t sparse_reorder_amd_qg(const SparseMatrix *A, idx_t *perm);

#endif /* SPARSE_REORDER_AMD_QG_INTERNAL_H */
