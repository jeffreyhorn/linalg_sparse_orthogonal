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
 * @brief Perform in-place LU factorization on a LuCsr matrix.
 *
 * Implements Gaussian elimination with partial pivoting using a
 * scatter-gather pattern on CSR arrays. L entries (multipliers) are stored
 * below the diagonal; U entries (including the diagonal) are stored on and
 * above. Row swaps are tracked in the piv_perm output array.
 *
 * The scatter-gather pattern works as follows:
 * 1. Scatter the pivot row into a dense workspace array
 * 2. For each row below the pivot with a nonzero in the pivot column:
 *    - Compute the multiplier
 *    - Scatter the row into a second dense workspace
 *    - Subtract the scaled pivot row
 *    - Gather surviving nonzeros back into CSR
 * 3. Gather the pivot row back from workspace
 *
 * Fill-in is handled by dynamically growing the CSR arrays when capacity
 * is exceeded. Entries smaller than drop_tol * |pivot| are dropped.
 *
 * @param csr       The LuCsr to factor in-place. On output, contains L\U.
 * @param tol       Absolute pivot tolerance. If the best pivot magnitude
 *                  is below this value, SPARSE_ERR_SINGULAR is returned.
 * @param drop_tol  Drop tolerance for fill-in. Entries with |value| <
 *                  drop_tol * |pivot| are dropped. Use 1e-14 as default.
 * @param[out] piv_perm  Row pivot permutation array (length n). On output,
 *                       piv_perm[k] = the original row index that was
 *                       swapped into position k. Caller must allocate.
 *                       May be NULL if pivot tracking is not needed.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if csr is NULL.
 * @return SPARSE_ERR_SINGULAR if a zero or near-zero pivot is encountered.
 * @return SPARSE_ERR_ALLOC if memory reallocation for fill-in fails.
 */
sparse_err_t lu_csr_eliminate(LuCsr *csr, double tol, double drop_tol, idx_t *piv_perm);

/**
 * @brief Free a LuCsr structure and its arrays.
 * @param csr  The LuCsr structure to free. Safe to call with NULL.
 */
void lu_csr_free(LuCsr *csr);

#endif /* SPARSE_LU_CSR_H */
