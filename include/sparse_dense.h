#ifndef SPARSE_DENSE_H
#define SPARSE_DENSE_H

/**
 * @file sparse_dense.h
 * @brief Dense matrix utilities for small inner computations.
 *
 * Provides a lightweight dense matrix type and basic operations (create,
 * free, multiply) used by eigenvalue solvers, bidiagonalization, and SVD
 * inner loops. Not intended for large-scale dense linear algebra.
 *
 * Storage is column-major: element (i,j) is at data[j * rows + i].
 */

#include "sparse_types.h"

/**
 * @brief Dense matrix stored in column-major order.
 */
typedef struct {
    double *data; /**< Column-major data: element (i,j) = data[j * rows + i] */
    idx_t rows;   /**< Number of rows */
    idx_t cols;   /**< Number of columns */
} dense_matrix_t;

/**
 * @brief Create a zero-initialized dense matrix.
 *
 * @param rows  Number of rows.
 * @param cols  Number of columns.
 * @return A new dense matrix, or NULL on allocation failure or invalid dims.
 */
dense_matrix_t *dense_create(idx_t rows, idx_t cols);

/**
 * @brief Free a dense matrix.
 *
 * @param M  The matrix to free. Safe to call with NULL.
 */
void dense_free(dense_matrix_t *M);

/**
 * @brief Dense matrix-matrix multiply: C = A * B.
 *
 * Requires A.cols == B.rows and C must be pre-allocated with dimensions
 * A.rows × B.cols. C is overwritten (not accumulated).
 *
 * @param A  Left operand (m × k).
 * @param B  Right operand (k × n).
 * @param C  Output (m × n). Must be pre-allocated.
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 * @return SPARSE_ERR_SHAPE if dimensions are incompatible.
 */
sparse_err_t dense_gemm(const dense_matrix_t *A, const dense_matrix_t *B,
                         dense_matrix_t *C);

/**
 * @brief Dense matrix-vector multiply: y = A * x.
 *
 * @param A  Matrix (m × n).
 * @param x  Input vector of length n.
 * @param y  Output vector of length m (overwritten).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if any argument is NULL.
 */
sparse_err_t dense_gemv(const dense_matrix_t *A, const double *x, double *y);

/**
 * @brief Access element (i,j) of a dense matrix (column-major).
 */
#define DENSE_AT(M, i, j) ((M)->data[(size_t)(j) * (size_t)(M)->rows + (size_t)(i)])

#endif /* SPARSE_DENSE_H */
