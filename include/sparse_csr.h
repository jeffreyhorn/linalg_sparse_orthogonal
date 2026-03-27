#ifndef SPARSE_CSR_H
#define SPARSE_CSR_H

/**
 * @file sparse_csr.h
 * @brief Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) formats.
 *
 * Provides conversion between the orthogonal linked-list representation
 * and standard compressed formats for interoperability with other libraries.
 *
 * @note These operate in physical index space. Do not use on matrices with
 *       non-identity permutations (e.g., after LU factorization).
 */

#include "sparse_matrix.h"

/**
 * @brief Compressed Sparse Row representation.
 *
 * row_ptr[i] .. row_ptr[i+1]-1 index into col_idx/values for row i.
 * row_ptr has length rows+1, col_idx and values have length nnz.
 */
typedef struct {
    idx_t rows;     /**< Number of rows */
    idx_t cols;     /**< Number of columns */
    idx_t nnz;      /**< Number of nonzeros */
    idx_t *row_ptr; /**< Row pointers (length rows+1) */
    idx_t *col_idx; /**< Column indices (length nnz) */
    double *values; /**< Nonzero values (length nnz) */
} SparseCsr;

/**
 * @brief Compressed Sparse Column representation.
 *
 * col_ptr[j] .. col_ptr[j+1]-1 index into row_idx/values for column j.
 * col_ptr has length cols+1, row_idx and values have length nnz.
 */
typedef struct {
    idx_t rows;     /**< Number of rows */
    idx_t cols;     /**< Number of columns */
    idx_t nnz;      /**< Number of nonzeros */
    idx_t *col_ptr; /**< Column pointers (length cols+1) */
    idx_t *row_idx; /**< Row indices (length nnz) */
    double *values; /**< Nonzero values (length nnz) */
} SparseCsc;

/**
 * @brief Convert an orthogonal linked-list matrix to CSR format.
 *
 * @param mat      Input matrix (not modified).
 * @param[out] csr Pointer to receive the CSR structure. Caller must free
 *                 with sparse_csr_free(). Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if mat or csr is NULL.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_to_csr(const SparseMatrix *mat, SparseCsr **csr);

/**
 * @brief Create an orthogonal linked-list matrix from CSR format.
 *
 * @param csr      Input CSR structure (not modified).
 * @param[out] mat Pointer to receive the new SparseMatrix. Caller must free
 *                 with sparse_free(). Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if csr or mat is NULL.
 * @return SPARSE_ERR_BADARG if CSR structure is invalid.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_from_csr(const SparseCsr *csr, SparseMatrix **mat);

/**
 * @brief Convert an orthogonal linked-list matrix to CSC format.
 *
 * @param mat      Input matrix (not modified).
 * @param[out] csc Pointer to receive the CSC structure. Caller must free
 *                 with sparse_csc_free(). Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if mat or csc is NULL.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_to_csc(const SparseMatrix *mat, SparseCsc **csc);

/**
 * @brief Create an orthogonal linked-list matrix from CSC format.
 *
 * @param csc      Input CSC structure (not modified).
 * @param[out] mat Pointer to receive the new SparseMatrix. Caller must free
 *                 with sparse_free(). Set to NULL on error.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if csc or mat is NULL.
 * @return SPARSE_ERR_BADARG if CSC structure is invalid.
 * @return SPARSE_ERR_ALLOC if memory allocation fails.
 */
sparse_err_t sparse_from_csc(const SparseCsc *csc, SparseMatrix **mat);

/** @brief Free a CSR structure and its arrays. */
void sparse_csr_free(SparseCsr *csr);

/** @brief Free a CSC structure and its arrays. */
void sparse_csc_free(SparseCsc *csc);

#endif /* SPARSE_CSR_H */
