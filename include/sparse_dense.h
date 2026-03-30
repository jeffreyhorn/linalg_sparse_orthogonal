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
sparse_err_t dense_gemm(const dense_matrix_t *A, const dense_matrix_t *B, dense_matrix_t *C);

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

/* ═══════════════════════════════════════════════════════════════════════
 * Givens rotations
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Compute Givens rotation that zeroes the second component.
 *
 * Computes c and s such that:
 *   [c  s] * [a] = [r]
 *   [-s c]   [b]   [0]
 * where r = hypot(a, b).
 *
 * @param a   First component.
 * @param b   Second component (to be zeroed).
 * @param c   Output: cosine of rotation angle.
 * @param s   Output: sine of rotation angle.
 */
void givens_compute(double a, double b, double *c, double *s);

/**
 * @brief Apply Givens rotation from the left to two row vectors.
 *
 * Replaces (x[k], y[k]) with (c*x[k] + s*y[k], -s*x[k] + c*y[k])
 * for k = 0..n-1.
 *
 * @param c  Cosine of rotation.
 * @param s  Sine of rotation.
 * @param x  First row vector (length n), modified in-place.
 * @param y  Second row vector (length n), modified in-place.
 * @param n  Vector length.
 */
void givens_apply_left(double c, double s, double *x, double *y, idx_t n);

/**
 * @brief Apply Givens rotation from the right to two column vectors.
 *
 * Applies G^T from the right: replaces (x[k], y[k]) with
 * (c*x[k] - s*y[k], s*x[k] + c*y[k]) for k = 0..n-1.
 *
 * @param c  Cosine of rotation.
 * @param s  Sine of rotation.
 * @param x  First column vector (length n), modified in-place.
 * @param y  Second column vector (length n), modified in-place.
 * @param n  Vector length.
 */
void givens_apply_right(double c, double s, double *x, double *y, idx_t n);

/* ═══════════════════════════════════════════════════════════════════════
 * 2×2 symmetric eigenvalue solver
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Compute eigenvalues of a 2×2 symmetric matrix.
 *
 * Solves for the eigenvalues of [[a, b], [b, d]].
 * Returns lambda1 <= lambda2 (ascending order).
 *
 * @param a        Element (0,0).
 * @param b        Element (0,1) = (1,0).
 * @param d        Element (1,1).
 * @param lambda1  Output: smaller eigenvalue.
 * @param lambda2  Output: larger eigenvalue.
 */
void eigen2x2(double a, double b, double d, double *lambda1, double *lambda2);

/* ═══════════════════════════════════════════════════════════════════════
 * Symmetric tridiagonal eigenvalue solver
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * @brief Compute all eigenvalues of a symmetric tridiagonal matrix.
 *
 * Uses the implicit QR algorithm with Wilkinson shifts and deflation.
 * The tridiagonal matrix T has diagonal entries diag[0..n-1] and
 * subdiagonal entries subdiag[0..n-2] (T is symmetric, so the
 * superdiagonal equals the subdiagonal).
 *
 * @param diag       Diagonal entries (length n). Overwritten with eigenvalues
 *                   in ascending order on success.
 * @param subdiag    Subdiagonal entries (length n-1). Destroyed on output.
 * @param n          Matrix dimension.
 * @param max_iter   Maximum total QR iterations (0 for default: 30*n).
 * @return SPARSE_OK on success.
 * @return SPARSE_ERR_NULL if diag or subdiag is NULL (when n > 1).
 * @return SPARSE_ERR_NOT_CONVERGED if max_iter reached without convergence.
 */
sparse_err_t tridiag_qr_eigenvalues(double *diag, double *subdiag, idx_t n, idx_t max_iter);

#endif /* SPARSE_DENSE_H */
