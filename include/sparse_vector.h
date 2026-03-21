#ifndef SPARSE_VECTOR_H
#define SPARSE_VECTOR_H

/**
 * @file sparse_vector.h
 * @brief Dense vector utility functions.
 *
 * Small helpers for BLAS-like operations on caller-allocated arrays.
 * Used by the LU solver, iterative refinement, and benchmarks.
 * All functions are NULL-safe (no-op on NULL inputs, return 0.0 where applicable).
 */

#include "sparse_types.h"

/**
 * @brief Compute the Euclidean (2-) norm: sqrt(sum(v[i]^2)).
 *
 * @param v  Input vector (NULL returns 0.0).
 * @param n  Vector length (n <= 0 returns 0.0).
 * @return The 2-norm of v.
 */
double vec_norm2(const double *v, idx_t n);

/**
 * @brief Compute the infinity norm: max(|v[i]|).
 *
 * @param v  Input vector (NULL returns 0.0).
 * @param n  Vector length.
 * @return The infinity norm of v.
 */
double vec_norminf(const double *v, idx_t n);

/**
 * @brief AXPY operation: y[i] += a * x[i] for i = 0..n-1.
 *
 * @param a  Scalar multiplier.
 * @param x  Input vector (NULL is a no-op).
 * @param y  Input/output vector (NULL is a no-op).
 * @param n  Vector length.
 */
void vec_axpy(double a, const double *x, double *y, idx_t n);

/**
 * @brief Copy: dst[i] = src[i] for i = 0..n-1.
 *
 * @param src  Source vector (NULL is a no-op).
 * @param dst  Destination vector (NULL is a no-op).
 * @param n    Vector length.
 */
void vec_copy(const double *src, double *dst, idx_t n);

/**
 * @brief Zero fill: v[i] = 0 for i = 0..n-1.
 *
 * @param v  Vector to zero (NULL is a no-op).
 * @param n  Vector length.
 */
void vec_zero(double *v, idx_t n);

/**
 * @brief Dot product: sum(x[i] * y[i]) for i = 0..n-1.
 *
 * @param x  First input vector (NULL returns 0.0).
 * @param y  Second input vector (NULL returns 0.0).
 * @param n  Vector length (n <= 0 returns 0.0).
 * @return The dot product of x and y.
 */
double vec_dot(const double *x, const double *y, idx_t n);

#endif /* SPARSE_VECTOR_H */
