#ifndef SPARSE_REORDER_H
#define SPARSE_REORDER_H

/**
 * @file sparse_reorder.h
 * @brief Fill-reducing reordering algorithms for sparse matrices.
 *
 * Provides three ordering algorithms:
 * - **RCM** (Reverse Cuthill-McKee): bandwidth reduction for banded systems.
 *   Symmetric permutation, square matrices only.
 * - **AMD** (Approximate Minimum Degree): symmetric fill reduction for
 *   Cholesky/LDL^T/LU. Symmetric permutation, square matrices only.
 * - **COLAMD** (Column Approximate Minimum Degree): column fill reduction
 *   for unsymmetric/QR problems. Column permutation, handles rectangular
 *   matrices.
 *
 * RCM and AMD compute symmetric permutations (P*A*P^T). COLAMD computes
 * a column-only permutation suitable for QR or column-pivoted LU.
 *
 * **Symmetric reordering (RCM/AMD) — square matrices:**
 * @code
 *   idx_t *perm = malloc(n * sizeof(idx_t));
 *   sparse_reorder_amd(A, perm);            // or sparse_reorder_rcm
 *   SparseMatrix *PA;
 *   sparse_permute(A, perm, perm, &PA);     // symmetric: same perm for rows+cols
 * @endcode
 *
 * **Column ordering (COLAMD) — rectangular OK:**
 * @code
 *   idx_t *perm = malloc(ncols * sizeof(idx_t));
 *   sparse_reorder_colamd(A, perm);         // column-only permutation
 *   // Apply as column permutation only (e.g., via QR opts)
 *   sparse_qr_opts_t opts = { .reorder = SPARSE_REORDER_COLAMD };
 *   sparse_qr_factor_opts(A, &opts, &qr);  // COLAMD applied internally
 * @endcode
 *
 * The permutation array perm[] maps new indices to old indices:
 * perm[new_i] = old_i.
 */

#include "sparse_matrix.h"

/**
 * @brief Compute a Reverse Cuthill-McKee ordering.
 *
 * Uses BFS on the symmetrized adjacency graph (A + A^T) to produce a
 * bandwidth-reducing permutation. Handles disconnected graphs by processing
 * each connected component separately. Uses a pseudo-peripheral starting
 * node heuristic for better results.
 *
 * @param A       Input matrix (must be square, not modified).
 * @param[out] perm  Permutation array of length n. On output, perm[new_i] = old_i.
 *                   Must be pre-allocated by the caller.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or perm is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 */
sparse_err_t sparse_reorder_rcm(const SparseMatrix *A, idx_t *perm);

/**
 * @brief Compute an Approximate Minimum Degree ordering.
 *
 * Minimum-degree ordering on the symmetrized adjacency graph (A + A^T),
 * implemented with bitset adjacency. At each step, eliminates the node with
 * the smallest degree (exact, via popcount), merges its neighbors' adjacency
 * sets to model fill-in, and updates degrees. Generally produces better
 * fill-in reduction than RCM for unstructured matrices, at higher cost.
 *
 * @param A       Input matrix (must be square, not modified).
 * @param[out] perm  Permutation array of length n. On output, perm[new_i] = old_i.
 *                   Must be pre-allocated by the caller.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or perm is NULL.
 * @return SPARSE_ERR_SHAPE if A is not square.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 */
sparse_err_t sparse_reorder_amd(const SparseMatrix *A, idx_t *perm);

/**
 * @brief Compute a Column Approximate Minimum Degree (COLAMD) ordering.
 *
 * Computes a column permutation that reduces fill-in during QR or LU
 * factorization of unsymmetric matrices. Operates on the column
 * adjacency graph (columns i and j are adjacent if they share a nonzero
 * row) without forming A^T*A explicitly. Dense rows are skipped to
 * control cost.
 *
 * Unlike AMD and RCM, COLAMD handles rectangular matrices (m != n)
 * and is designed specifically for unsymmetric column structure.
 *
 * @param A       Input matrix (may be rectangular, not modified).
 * @param[out] perm  Column permutation array of length ncols.
 *                   On output, perm[new_j] = old_j.
 *                   Must be pre-allocated by the caller.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if A or perm is NULL.
 * @return SPARSE_ERR_ALLOC if workspace allocation fails.
 */
sparse_err_t sparse_reorder_colamd(const SparseMatrix *A, idx_t *perm);

/**
 * @brief Apply a row and column permutation to create a reordered matrix.
 *
 * Creates a new matrix B where B(i,j) = A(row_perm[i], col_perm[j])
 * in physical index space. For symmetric reordering (fill-reducing),
 * pass the same array for both row_perm and col_perm.
 *
 * @note Operates in physical index space. Intended for use on unfactored
 *       matrices with identity permutations. Do not use on matrices that
 *       have been LU-factored (non-identity row_perm/col_perm).
 *
 * @param A         Input matrix (not modified).
 * @param row_perm  Row permutation array of length A->rows. row_perm[new_i] = old_i.
 * @param col_perm  Column permutation array of length A->cols. col_perm[new_j] = old_j.
 * @param[out] B    Pointer to receive the reordered matrix. Caller must free with
 *                  sparse_free(). Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_BADARG if row_perm or col_perm is not a valid permutation
 *         (out-of-range index or duplicate entry).
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_permute(const SparseMatrix *A, const idx_t *row_perm, const idx_t *col_perm,
                            SparseMatrix **B);

/**
 * @brief Compute the bandwidth of a sparse matrix.
 *
 * The bandwidth is max |i - j| over all nonzero entries a_ij.
 * Useful for quantifying the effectiveness of bandwidth-reducing
 * reorderings like RCM.
 *
 * @note Operates in physical index space. Do not use on matrices with
 *       non-identity permutations (e.g., after LU factorization).
 *
 * @param A  Input matrix (not modified).
 * @return The bandwidth, or 0 if A is NULL or empty.
 */
idx_t sparse_bandwidth(const SparseMatrix *A);

#endif /* SPARSE_REORDER_H */
