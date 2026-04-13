#ifndef SPARSE_ANALYSIS_INTERNAL_H
#define SPARSE_ANALYSIS_INTERNAL_H

/**
 * @file sparse_analysis_internal.h
 * @brief Internal data structures for symbolic analysis (etree, column counts).
 *
 * Not part of the public API. Used by sparse_etree.c and sparse_analysis.c.
 */

#include "sparse_matrix_internal.h"
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Elimination tree
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Compute the elimination tree of a symmetric sparse matrix.
 *
 * Uses Liu's algorithm with path compression (union-find). The etree
 * parent of column j is the smallest column index k > j such that L(k,j)
 * is nonzero in the Cholesky factor. For root nodes, parent[j] = -1.
 *
 * Only the lower triangular part of A (including diagonal) is used.
 *
 * @param A       Symmetric sparse matrix (lower triangle used).
 * @param parent  Output array of length n. parent[j] = etree parent of j,
 *                or -1 if j is a root.
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_etree_compute(const SparseMatrix *A, idx_t *parent);

/**
 * Compute a postorder traversal of the elimination tree.
 *
 * The postorder visits children before parents, which is the natural
 * bottom-up order for symbolic and numeric factorization.
 *
 * @param parent     Etree parent array (length n, -1 for roots).
 * @param n          Matrix dimension.
 * @param postorder  Output array of length n. postorder[k] = column visited
 *                   at position k in the postorder.
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_etree_postorder(const idx_t *parent, idx_t n, idx_t *postorder);

/**
 * Compute the exact column counts of the Cholesky factor L.
 *
 * colcount[j] = number of nonzeros in column j of L, including the diagonal.
 * The total nnz(L) = sum of all column counts.
 *
 * Uses a bottom-up traversal of the elimination tree: for each column j
 * (in postorder), the row indices of L(:,j) are the union of the original
 * lower-triangle entries in column j and the row sets propagated up from
 * children of j in the etree (excluding row j itself).
 *
 * Only the lower triangular part of A (including diagonal) is used.
 *
 * @param A          Symmetric sparse matrix (lower triangle used).
 * @param parent     Etree parent array (length n, -1 for roots).
 * @param postorder  Etree postorder traversal (length n).
 * @param colcount   Output array of length n.
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_colcount(const SparseMatrix *A, const idx_t *parent, const idx_t *postorder,
                             idx_t *colcount);

/* ═══════════════════════════════════════════════════════════════════════
 * Symbolic factorization structure
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Compressed-column symbolic structure of a triangular factor.
 *
 * col_ptr[j]..col_ptr[j+1]-1 index into row_idx for the row indices
 * of nonzeros in column j. Row indices within each column are sorted
 * in ascending order.
 */
typedef struct {
    idx_t *col_ptr; /**< Column pointers (length n+1). */
    idx_t *row_idx; /**< Row indices (length nnz). */
    idx_t n;        /**< Matrix dimension. */
    idx_t nnz;      /**< Total nonzeros. */
} sparse_symbolic_t;

/**
 * Compute the exact symbolic structure of the Cholesky factor L.
 *
 * For each column j (processed in postorder), the row indices of L(:,j)
 * are the union of the lower-triangle entries of A(:,j) and the row
 * sets propagated from children of j in the elimination tree.
 *
 * The resulting structure is stored in compressed-column format in sym.
 * Row indices within each column are sorted in ascending order.
 *
 * @param A          Symmetric sparse matrix (lower triangle used).
 * @param parent     Etree parent array (length n, -1 for roots).
 * @param postorder  Etree postorder traversal (length n).
 * @param colcount   Column counts of L (from sparse_colcount).
 * @param sym        Output symbolic structure (caller frees with
 *                   sparse_symbolic_free).
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_symbolic_cholesky(const SparseMatrix *A, const idx_t *parent,
                                      const idx_t *postorder, const idx_t *colcount,
                                      sparse_symbolic_t *sym);

/**
 * Compute upper-bound symbolic structure for LU factorization.
 *
 * For an unsymmetric matrix A, the exact L and U sparsity depends on
 * pivoting, so we compute a superset. The approach: build the column
 * interaction graph (the sparsity structure of A^T * A) as an explicit
 * sparse matrix, then compute its elimination tree and symbolic
 * Cholesky. The resulting structure is a valid upper bound for both L
 * and U columns.
 *
 * Cost: O(sum_i row_nnz(i)^2) time and memory for building A^T*A,
 * which can be expensive for matrices with very dense rows.
 *
 * If perm is non-NULL, it is applied as a symmetric permutation
 * (fill-reducing reordering) before computing the symbolic structure.
 *
 * sym_L receives the lower-triangle bound (including diagonal).
 * sym_U receives the upper-triangle bound (including diagonal).
 * Either may be NULL if not needed.
 *
 * @pre sym_L and sym_U (when non-NULL) must be zeroed or previously
 *      freed via sparse_symbolic_free() before calling. Passing a
 *      struct with live allocations will leak memory.
 *
 * @param A      Sparse matrix (may be unsymmetric).
 * @param perm   Fill-reducing permutation (length n), or NULL for natural.
 * @param sym_L  Output lower-triangle bound, or NULL. Must be zeroed.
 * @param sym_U  Output upper-triangle bound, or NULL. Must be zeroed.
 * @return SPARSE_OK on success.
 */
sparse_err_t sparse_symbolic_lu(const SparseMatrix *A, const idx_t *perm, sparse_symbolic_t *sym_L,
                                sparse_symbolic_t *sym_U);

/**
 * Free a symbolic structure's internal arrays.
 * Safe to call on a zeroed struct (no-op).
 */
void sparse_symbolic_free(sparse_symbolic_t *sym);

#endif /* SPARSE_ANALYSIS_INTERNAL_H */
