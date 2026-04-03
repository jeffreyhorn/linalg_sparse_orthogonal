#ifndef SPARSE_LU_CSR_H
#define SPARSE_LU_CSR_H

/**
 * @file sparse_lu_csr.h
 * @brief CSR working format for LU factorization.
 *
 * Provides a CSR representation designed specifically for LU elimination.
 * Unlike the general SparseCsr (sparse_csr.h), this format:
 * - Works in logical index space (applies row/col permutations on conversion)
 * - Pre-allocates extra capacity for fill-in during elimination
 * - Includes a dense workspace row for the scatter-gather elimination pattern
 *
 * Conversion pipeline: SparseMatrix → LuCsr → eliminate → LuCsr → SparseMatrix
 */

#include "sparse_matrix.h"

/**
 * @brief CSR working format for LU elimination.
 *
 * Stores the matrix in compressed sparse row format with extra capacity
 * for fill-in entries created during Gaussian elimination. The row_ptr,
 * col_idx, and values arrays are in logical index order.
 */
typedef struct {
    idx_t n;        /**< Matrix dimension (square n×n) */
    idx_t nnz;      /**< Current number of stored nonzeros */
    idx_t capacity; /**< Allocated length of col_idx/values arrays */
    idx_t *row_ptr; /**< Row pointers (length n+1). row_ptr[i]..row_ptr[i+1]-1
                         index into col_idx/values for logical row i. */
    idx_t *col_idx; /**< Column indices in logical space (length capacity) */
    double *values; /**< Nonzero values (length capacity) */
} LuCsr;

/**
 * @brief Convert a SparseMatrix to CSR working format for LU elimination.
 *
 * Reads the matrix in logical index order (applying row_perm/col_perm)
 * and stores entries sorted by logical column within each logical row.
 * Allocates extra capacity (fill_factor × nnz) for fill-in during
 * elimination.
 *
 * @param mat         Input matrix (not modified).
 * @param fill_factor Capacity multiplier for fill-in (e.g., 2.0 allocates
 *                    2× the initial nnz). Clamped to [1.0, 20.0].
 * @param[out] csr    Pointer to receive the LuCsr structure. Caller must
 *                    free with lu_csr_free(). Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if mat or csr is NULL.
 * @return SPARSE_ERR_SHAPE if mat is not square.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t lu_csr_from_sparse(const SparseMatrix *mat, double fill_factor, LuCsr **csr);

/**
 * @brief Convert a LuCsr back to a SparseMatrix (linked-list format).
 *
 * Creates a new SparseMatrix from the CSR data. The resulting matrix
 * has identity permutations (logical = physical). Entries with zero
 * value are skipped.
 *
 * @param csr         Input LuCsr structure (not modified).
 * @param[out] mat    Pointer to receive the new SparseMatrix. Caller must
 *                    free with sparse_free(). Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if csr or mat is NULL.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t lu_csr_to_sparse(const LuCsr *csr, SparseMatrix **mat);

/**
 * @brief Free a LuCsr structure and its arrays.
 * @param csr  The LuCsr structure to free. Safe to call with NULL.
 */
void lu_csr_free(LuCsr *csr);

#endif /* SPARSE_LU_CSR_H */
