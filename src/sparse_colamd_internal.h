#ifndef SPARSE_COLAMD_INTERNAL_H
#define SPARSE_COLAMD_INTERNAL_H

/**
 * @file sparse_colamd_internal.h
 * @brief Internal data structures and helpers for COLAMD ordering.
 *
 * COLAMD (Column Approximate Minimum Degree) computes a fill-reducing
 * column permutation for unsymmetric matrices by operating on the column
 * adjacency graph: columns i and j are adjacent if they share a nonzero
 * row in A. This is equivalent to computing minimum degree on A^T*A
 * without forming it explicitly.
 *
 * Not part of the public API.
 */

#include "sparse_matrix_internal.h"
#include <stdint.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Column adjacency graph (CSR format)
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Column adjacency graph in CSR format.
 *
 * col_adj_ptr[j]..col_adj_ptr[j+1]-1 index into col_adj_list for the
 * column indices adjacent to column j (sharing a nonzero row).
 * Self-loops (j adjacent to j) are excluded.
 */
typedef struct {
    idx_t *col_adj_ptr;  /**< Column pointers (length ncols+1) */
    idx_t *col_adj_list; /**< Adjacent column indices */
    idx_t ncols;         /**< Number of columns */
    idx_t nnz_adj;       /**< Total adjacency entries */
} colamd_graph_t;

/**
 * Build the column adjacency graph from A's row structure.
 *
 * Two columns i and j are adjacent if there exists a row k such that
 * A(k,i) != 0 and A(k,j) != 0. This is the sparsity pattern of A^T*A
 * (without diagonal), computed without forming A^T*A explicitly.
 *
 * Dense rows (rows with more than dense_threshold nonzeros) are skipped
 * to avoid O(nnz_row^2) blowup. Pass dense_threshold <= 0 to include
 * all rows.
 *
 * @param A                The sparse matrix.
 * @param dense_threshold  Rows with nnz > this are skipped. Use <= 0
 *                         to include all rows.
 * @param graph            Output graph (caller frees with colamd_graph_free).
 * @return SPARSE_OK on success.
 */
sparse_err_t colamd_build_graph(const SparseMatrix *A, idx_t dense_threshold,
                                colamd_graph_t *graph);

/**
 * Free a column adjacency graph.
 * Safe to call on a zeroed struct (no-op).
 */
void colamd_graph_free(colamd_graph_t *graph);

/* ═══════════════════════════════════════════════════════════════════════
 * COLAMD ordering
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Compute a column approximate minimum degree ordering.
 *
 * Builds the column adjacency graph from A (skipping dense rows), then
 * performs minimum degree elimination on that graph to produce a column
 * permutation that reduces fill-in during factorization.
 *
 * @param A     The sparse matrix (may be rectangular).
 * @param perm  Output permutation array (length ncols). perm[new] = old.
 * @return SPARSE_OK on success.
 */
sparse_err_t colamd_order(const SparseMatrix *A, idx_t *perm);

#endif /* SPARSE_COLAMD_INTERNAL_H */
