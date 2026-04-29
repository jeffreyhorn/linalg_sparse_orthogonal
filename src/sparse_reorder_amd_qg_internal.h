#ifndef SPARSE_REORDER_AMD_QG_INTERNAL_H
#define SPARSE_REORDER_AMD_QG_INTERNAL_H

/**
 * @file sparse_reorder_amd_qg_internal.h
 * @brief Sprint 22 Days 10-12 quotient-graph AMD — internal API.
 *
 * Replaces the bitset AMD in `src/sparse_reorder.c:421`
 * (`sparse_reorder_amd`) with a quotient-graph implementation
 * (Amestoy/Davis/Duff 2004) that runs in O(nnz) memory.  Day 10
 * lands the design + stub; Day 11 fills in the elimination loop;
 * Day 12 swaps the public `sparse_reorder_amd` body to call this
 * helper.  Until Day 12, this file is internal-only and exists so
 * the Day-11 tests can drive the new implementation alongside the
 * (still-production) bitset path.
 *
 * Contract: same signature + same permutation convention as
 * `sparse_reorder_amd` — `perm[i]` is the i-th vertex eliminated.
 * Pivot tie-breaking is allowed to differ from the bitset version,
 * so the validation strategy on Day 11 is **equivalent fill** (nnz
 * of symbolic Cholesky), not bit-identical permutations.
 */

#include "sparse_matrix.h"
#include "sparse_types.h"

/**
 * @brief Quotient-graph Approximate Minimum Degree ordering.
 *
 * Same input/output contract as `sparse_reorder_amd` from
 * `include/sparse_reorder.h`, but uses a single integer workspace
 * (≈ 5·nnz + 6·n + 1 entries) instead of an n×n bitset.  Day 10
 * stub returns SPARSE_ERR_BADARG; Day 11 ships the full
 * implementation.
 *
 * @param A     Input matrix (must be square, not modified).
 * @param perm  Output permutation array, length n; `perm[new_i] =
 *              old_i`.  Caller pre-allocates.
 *
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or perm is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC on allocation failure.
 * @return SPARSE_ERR_BADARG (Day 10 stub only).
 */
sparse_err_t sparse_reorder_amd_qg(const SparseMatrix *A, idx_t *perm);

#endif /* SPARSE_REORDER_AMD_QG_INTERNAL_H */
